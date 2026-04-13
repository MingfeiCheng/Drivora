import os
import copy
import json
import time
import pickle
import shutil
import signal
import threading
import traceback
import concurrent
import subprocess
import numpy as np
import multiprocessing as mp
import select

from deap import base, creator, tools
from datetime import datetime
from loguru import logger
from typing import Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field, asdict, fields

from scenario_corpus.openscenario.config import ScenarioConfig
from scenario_runner.config import GlobalConfig
from scenario_runner.ctn_manager import create_ctn_manager, CtnConfig, CtnSimOperator
from tools.recorder_tool import load_observation, load_runtime_result, visualize_trajectories


def _progress_bar(current, total, length=20):
    ratio = min(max(current / total, 0), 1)
    filled = int(ratio * length)
    bar = "▓" * filled + "░" * (length - filled)
    return bar, ratio


# ──────────────────────────────────────────────────────────────────────
# FuzzSeed — data container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FuzzSeed:
    """Data container for fuzzing seeds."""
    id: str
    scenario: ScenarioConfig
    oracle_result: Dict[str, Any] = field(default_factory=dict)
    feedback_result: Dict[str, Any] = field(default_factory=dict)
    sample_result: Dict[str, Any] = field(default_factory=dict)
    is_expected: bool = False
    is_ignored: bool = False
    scenario_dir: str = ""

    def set_id(self, new_id: str):
        self.id = new_id
        self.scenario.id = new_id

    def set_scenario_dir(self, scenario_dir: str):
        self.scenario_dir = scenario_dir

    @classmethod
    def load_from_scenario_file(cls, config_path: str) -> "FuzzSeed":
        with open(config_path, 'r') as f:
            data = json.load(f)
        scenario = ScenarioConfig.model_validate(data)
        return cls(id="init_seed", scenario=scenario)

    @classmethod
    def load_from_dict(cls, data: dict) -> "FuzzSeed":
        init_args = {}
        for f in fields(cls):
            if f.name == "scenario":
                init_args["scenario"] = ScenarioConfig.model_validate(data["scenario"])
            else:
                init_args[f.name] = data.get(f.name, getattr(cls, f.name, None))
        return cls(**init_args)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["scenario"] = self.scenario.model_dump()
        return result

    def to_deap_args(self) -> dict:
        result = asdict(self)
        result["scenario"] = self.scenario
        return result


# ──────────────────────────────────────────────────────────────────────
# Fuzzer — base class with all reusable infrastructure
# ──────────────────────────────────────────────────────────────────────

class Fuzzer:

    SCENARIO_ENTRY = "scenario_corpus.openscenario.scenario:OpenScenario"
    MANAGER_NAME = "default"

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        self.fuzzer_config = fuzzer_config
        self.agent_config = agent_config
        self.scenario_config = scenario_config

        self.resume = GlobalConfig.resume
        self.output_root = GlobalConfig.output_root

        # directories
        self.result_folder = os.path.join(self.output_root, 'results')
        self.tmp_dir = os.path.join(self.output_root, 'tmp')
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.tmp_dir, 'checkpoint.pkl')

        # save config snapshot
        OmegaConf.save(config=fuzzer_config, f=os.path.join(self.output_root, 'fuzzer_config.yaml'))

        # basic fuzzer config
        self.time_budget = fuzzer_config.get('time_budget', 1.0)
        self.seed_path = self.scenario_config.get('seed_path', None)

        # basic agent config
        self.agent_entry_point = self.agent_config.get('entry_point', None)
        if self.agent_entry_point is None:
            raise ValueError("Please provide a valid agent entry point.")
        self.agent_config_path = self.agent_config.get('config_path', {})

        # time accounting
        self.time_counter_file = os.path.join(self.tmp_dir, 'time_counter.txt')
        self.time_counter = 0.0
        if os.path.exists(self.time_counter_file):
            with open(self.time_counter_file, 'r') as f:
                line = f.readline()
                if line:
                    self.time_counter = float(line.rstrip())

        self._terminated_early = False
        if self.termination_check(datetime.now()):
            logger.info(f"Already tested for {self.time_budget} hours, skip.")
            self._terminated_early = True

        # load pipeline config
        fuzzer_config_path = fuzzer_config.get('config_path', None)
        if fuzzer_config_path is None or not os.path.isfile(fuzzer_config_path):
            raise ValueError(f"Invalid fuzzer config path: {fuzzer_config_path}")
        self.pipeline_config = OmegaConf.load(fuzzer_config_path)

        # container manager
        self.ctn_manager = create_ctn_manager(
            run_tag=GlobalConfig.run_tag,
            carla_image=GlobalConfig.carla_image,
            carla_fps=GlobalConfig.carla_fps,
            random_seed=GlobalConfig.carla_random_seed,
            is_sync=GlobalConfig.carla_is_sync,
        )

        # sub-configs (subclass creates actual objects from these)
        self.mutator_config = self.pipeline_config.get('mutator', {})
        self.feedback_config = self.pipeline_config.get('feedback', {})
        self.oracle_config = self.pipeline_config.get('oracle', {})

        # placeholders — subclass MUST set these
        self.mutator = None
        self.feedback = None
        self.oracle = None

        self.used_time = 0.0
        self.subprocess_pids = mp.Manager().list()

        # ── shared fuzzer state ──
        self.global_search_step = 0
        self.initialized = False
        self.seed_recorder = []
        self.F_corpus = []

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "fitnesses"]

        # DEAP toolbox
        self.toolbox = None

    # ──────────────────────────────────────────────────────────────
    # DEAP setup
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        """Default DEAP toolbox — registers evaluate and re_evaluate.
        Subclass can override or call super().setup_deap() then register more."""
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.execute_evaluate)
        self.toolbox.register("re_evaluate", self.re_evaluate_population)

    # ──────────────────────────────────────────────────────────────
    # Checkpoint save / load
    # ──────────────────────────────────────────────────────────────

    def _get_checkpoint_data(self) -> dict:
        """Return checkpoint dict. Subclass can override to add fields."""
        return {
            "global_search_step": self.global_search_step,
            "initialized": self.initialized,
            "seed_recorder": self.seed_recorder,
            "F_corpus": self.F_corpus,
        }

    def _restore_checkpoint_data(self, data: dict):
        """Restore from checkpoint dict. Subclass can override to read extra fields."""
        self.global_search_step = data.get('global_search_step', 0)
        self.initialized = data.get('initialized', False)
        self.seed_recorder = data.get('seed_recorder', [])
        self.F_corpus = data.get('F_corpus', [])

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            logger.warning('Checkpoint file not found, start from scratch.')
            return

        with open(self.checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self._restore_checkpoint_data(checkpoint_data)
        logger.info(f"Resumed from checkpoint: step={self.global_search_step}, "
                    f"F_corpus={len(self.F_corpus)}, seeds={len(self.seed_recorder)}")

        logbook_file = os.path.join(self.output_root, "logbook.json")
        if os.path.exists(logbook_file):
            with open(logbook_file, 'r') as f:
                for entry in json.load(f):
                    self.logbook.record(**entry)

        logger.info('Loaded checkpoint from {}', self.checkpoint_path)

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(self._get_checkpoint_data(), f)
        logger.info('Saved checkpoint to {}', self.checkpoint_path)

        # overview
        overview = {
            'summary': {
                'F_size': len(self.F_corpus),
                'time_budget_hours': self.time_budget,
                'time_used_hours': self.used_time / 3600.0,
                'F_corpus': self.F_corpus,
            },
            'details': {},
        }
        for sb in self.seed_recorder:
            overview['details'][sb['id']] = {
                'scenario_id': sb['id'],
                'is_expected': sb['is_expected'],
                'is_ignored': sb['is_ignored'],
                'oracle_result': sb['oracle_result'],
            }
        with open(os.path.join(self.output_root, 'overview.json'), 'w') as f:
            json.dump(overview, f, indent=4)

        with open(os.path.join(self.output_root, "logbook.json"), 'w') as f:
            json.dump(self.logbook, f, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────
    # Evaluation pipeline (oracle + feedback)
    # ──────────────────────────────────────────────────────────────

    def _make_error_oracle_result(self) -> dict:
        return {
            'expected': False,
            'ignored': True,
            'runtime_results': 'error',
            'criteria_summary': {},
            'offline_results': {},
        }

    def execute_evaluate(self, individuals: list):
        """Run scenarios, then evaluate oracle + feedback."""
        running_seeds = [ind[0] for ind in individuals]
        logger.info(f"Executing {len(running_seeds)} individuals ...")

        exec_results = self.execute_population(running_seeds)

        batch_info = {"seed_recorder": [], "F_corpus": []}

        for exec_res in exec_results:
            ind_index = exec_res['index']
            scenario_dir = exec_res['scenario_dir']

            if not exec_res['status']:
                feedback_result = self.feedback.get_default_feedback()
                oracle_result = self._make_error_oracle_result()
            else:
                try:
                    visualize_trajectories(scenario_dir)
                    scenario_observation = load_observation(scenario_dir)
                    runtime_oracle_results = load_runtime_result(scenario_dir)
                    oracle_result = self.oracle.evaluate(scenario_observation, runtime_oracle_results)
                    feedback_result = self.feedback.evaluate(scenario_observation, oracle_result)
                except Exception as e:
                    logger.warning(f"Oracle/feedback evaluation failed for {scenario_dir}: {e}")
                    feedback_result = self.feedback.get_default_feedback()
                    oracle_result = self._make_error_oracle_result()

            ind = individuals[ind_index][0]
            ind.oracle_result = oracle_result
            ind.is_expected = oracle_result['expected']
            ind.is_ignored = oracle_result['ignored']
            ind.feedback_result = feedback_result
            ind.set_scenario_dir(scenario_dir if scenario_dir else "")
            ind = self.assign_feedback_to_ind(ind, feedback_result)
            individuals[ind_index][0] = ind

            batch_info['seed_recorder'].append({
                'id': ind.id,
                'is_expected': ind.is_expected,
                'is_ignored': ind.is_ignored,
                'oracle_result': ind.oracle_result,
                'feedback_result': ind.feedback_result,
            })

            if ind.is_expected and not ind.is_ignored:
                batch_info['F_corpus'].append({
                    'id': ind.id,
                    'is_expected': ind.is_expected,
                    'is_ignored': ind.is_ignored,
                    'oracle_result': ind.oracle_result,
                })

        self.seed_recorder.extend(batch_info['seed_recorder'])
        self.F_corpus.extend(batch_info['F_corpus'])
        self.save_checkpoint()
        return individuals

    def re_evaluate_population(self, individuals: list):
        """Re-evaluate existing scenario results (no re-execution)."""
        logger.info(f"Re-evaluating {len(individuals)} individuals ...")

        for ind_index, ind in enumerate(individuals):
            ind[0].set_id(f"{ind[0].id}_reeval")
            scenario_dir = ind[0].scenario_dir

            try:
                scenario_observation = load_observation(scenario_dir)
                runtime_oracle_results = load_runtime_result(scenario_dir)
                oracle_result = self.oracle.evaluate(scenario_observation, runtime_oracle_results)
                feedback_result = self.feedback.evaluate(scenario_observation, oracle_result)
            except Exception as e:
                logger.error(f"Error re-evaluating {ind[0].id}: {e}")
                traceback.print_exc()
                feedback_result = self.feedback.get_default_feedback()
                oracle_result = self._make_error_oracle_result()

            inner = ind[0]
            inner.oracle_result = oracle_result
            inner.is_expected = oracle_result['expected']
            inner.is_ignored = oracle_result['ignored']
            inner.feedback_result = feedback_result
            inner = self.assign_feedback_to_ind(inner, feedback_result)
            individuals[ind_index][0] = inner

            self.seed_recorder.append({
                'id': inner.id,
                'is_expected': inner.is_expected,
                'is_ignored': inner.is_ignored,
                'oracle_result': inner.oracle_result,
                'feedback_result': inner.feedback_result,
            })

        return individuals

    # ──────────────────────────────────────────────────────────────
    # Feedback / fitness assignment — override in subclass
    # ──────────────────────────────────────────────────────────────

    def assign_feedback_to_ind(self, ind, feedback_result):
        """Assign fitness values to a DEAP individual from feedback.
        Must be overridden by subclass."""
        raise NotImplementedError

    @staticmethod
    def clone_ind(ind):
        new_seed = copy.deepcopy(ind[0])
        new_ind = creator.Individual([new_seed])
        if ind.fitness.valid:
            new_ind.fitness.values = ind.fitness.values
        return new_ind

    # ──────────────────────────────────────────────────────────────
    # Logbook
    # ──────────────────────────────────────────────────────────────

    def record_logbook(self, gen, pop):
        fitness_lst = []
        for item in pop:
            ind = item[0] if isinstance(item, list) else item
            if hasattr(ind, 'fitness') and ind.fitness.valid:
                fitness_lst.append(ind.fitness.values)

        if not fitness_lst:
            mean_fitness = [float('inf')]
        else:
            mean = np.mean(np.array(fitness_lst), axis=0)
            mean_fitness = [float(mean)] if np.isscalar(mean) else mean.tolist()

        self.logbook.record(gen=gen, fitnesses=mean_fitness)

    # ──────────────────────────────────────────────────────────────
    # Run lifecycle
    # ──────────────────────────────────────────────────────────────

    def run(self):
        if self._terminated_early:
            logger.info("Time budget exhausted — nothing to do.")
            return
        if self.resume:
            self.load_checkpoint()
        start_time = datetime.now()
        self._run(start_time)

    def _run(self, start_time):
        """Main fuzzing loop — must be overridden."""
        raise NotImplementedError

    def close(self):
        self.cleanup_all_subprocesses()

    # ──────────────────────────────────────────────────────────────
    # Time budget
    # ──────────────────────────────────────────────────────────────

    def termination_check(self, start_time) -> bool:
        t_delta = (datetime.now() - start_time).total_seconds()
        self.used_time = t_delta + self.time_counter
        with open(self.time_counter_file, 'w') as f:
            f.write(f"{self.used_time}\n")
        if self.time_budget is not None and self.used_time / 3600.0 > self.time_budget:
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # Process management
    # ──────────────────────────────────────────────────────────────

    def kill_process_tree(self, pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass

    def cleanup_all_subprocesses(self):
        logger.warning(f"Killing {len(self.subprocess_pids)} subprocesses...")
        for pid in list(self.subprocess_pids):
            self.kill_process_tree(pid)
        self.subprocess_pids[:] = []

    def restart_containers(self):
        """Stop and restart all CARLA containers for a clean state."""
        logger.warning("Restarting all CARLA containers...")
        for op_cfg in self.ctn_manager._all_ops:
            try:
                ctn = CtnSimOperator(
                    idx=op_cfg.idx,
                    container_name=op_cfg.container_name,
                    gpu=op_cfg.gpu,
                    random_seed=op_cfg.random_seed,
                    docker_image=op_cfg.docker_image,
                    fps=op_cfg.fps,
                    is_sync_mode=op_cfg.is_sync_mode,
                )
                ctn.stop()
                ctn.start()
                logger.info(f"Container {op_cfg.container_name} restarted.")
            except Exception as e:
                logger.error(f"Failed to restart {op_cfg.container_name}: {e}")

    # ──────────────────────────────────────────────────────────────
    # Scenario execution (subprocess)
    # ──────────────────────────────────────────────────────────────

    def execute_instance(
        self,
        venv_dir: str,
        scenario_execute_script: str,
        scenario_entry_point: str,
        scenario_config: object,
        ctn_config: object,
        scenario_dir: str = None,
        manager_name: str = "default",
        max_sim_time: float = 300,
        debug: bool = False,
        pytree_debug: bool = False,
        open_vis: bool = True,
        timeout: float = 900,
    ) -> Tuple[bool, str, int]:

        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
        os.makedirs(scenario_dir, exist_ok=True)

        scenario_json_path = os.path.join(scenario_dir, "scenario.json")
        ctn_json_path = os.path.join(scenario_dir, "ctn_config.json")

        with open(scenario_json_path, "w") as f:
            json.dump(scenario_config.model_dump(), f, indent=4)
        with open(ctn_json_path, "w") as f:
            json.dump(ctn_config.to_dict(), f, indent=4)

        # Use the python binary from the uv venv (resolve to absolute path)
        venv_python = os.path.abspath(os.path.join(venv_dir, "bin", "python"))
        scenario_execute_script = os.path.abspath(scenario_execute_script)
        if not os.path.isfile(venv_python):
            raise FileNotFoundError(f"Python not found in venv: {venv_python}")

        # Log file — subprocess stdout+stderr both go here
        eval_log_path = os.path.join(scenario_dir, 'eval.log')

        cmd = [
            venv_python, "-u", scenario_execute_script,
            "--scenario_entry_point", scenario_entry_point,
            "--scenario_config", scenario_json_path,
            "--ctn_config", ctn_json_path,
            "--scenario_dir", scenario_dir,
            "--manager_name", manager_name,
            "--max_sim_time", str(max_sim_time),
        ]
        if debug:
            cmd.append("--debug")
        if pytree_debug:
            cmd.append("--pytree_debug")
        if open_vis:
            cmd.append("--open_vis")
        if hasattr(GlobalConfig, 'save_agent_internal') and GlobalConfig.save_agent_internal:
            cmd.append("--save_agent_internal")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(ctn_config.gpu)

        f_log = open(eval_log_path, 'w')

        process = subprocess.Popen(
            cmd,
            stdout=f_log,
            stderr=subprocess.STDOUT,  # merge stderr into stdout -> single log file
            preexec_fn=os.setsid,
            env=env,
        )

        child_pid = process.pid
        self.subprocess_pids.append(child_pid)

        stall_timeout = 120.0  # kill if log has no update for this long
        last_log_mtime = time.time()

        start_ts = time.time()
        while True:
            elapsed = time.time() - start_ts

            # Hard timeout
            if elapsed > timeout:
                logger.error(f"TIMEOUT {elapsed:.1f}/{timeout}s. Killing PID={child_pid}")
                try:
                    os.killpg(os.getpgid(child_pid), signal.SIGKILL)
                except Exception:
                    pass
                break

            # Stall detection: kill if log file hasn't been updated
            try:
                cur_mtime = os.path.getmtime(eval_log_path)
                if cur_mtime > last_log_mtime:
                    last_log_mtime = cur_mtime
            except OSError:
                pass

            if time.time() - last_log_mtime > stall_timeout:
                logger.error(f"STALL detected: eval.log not updated for {stall_timeout}s. Killing PID={child_pid}")
                try:
                    os.killpg(os.getpgid(child_pid), signal.SIGKILL)
                except Exception:
                    pass
                break

            # Check if process finished
            if process.poll() is not None:
                logger.info(f"Process PID={child_pid} finished (rc={process.returncode}) in {elapsed:.1f}s.")
                break

            # Check if result files appeared (scenario finished but process still cleaning up)
            result_file = os.path.join(scenario_dir, 'result.json')
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                break

            # Progress bar
            try:
                bar, ratio = _progress_bar(elapsed, timeout)
                print(
                    f"\r[PID={child_pid} GPU={ctn_config.gpu} SID={scenario_config.id}] "
                    f"Running {bar} {elapsed:.1f}s / {timeout:.0f}s ({ratio*100:4.1f}%)",
                    end="", flush=True,
                )
            except BrokenPipeError:
                pass

            time.sleep(0.5)

        print()
        f_log.close()

        # Log exit info
        rc = process.returncode
        if rc is not None and rc < 0:
            sig_name = {-11: "SIGSEGV", -9: "SIGKILL", -6: "SIGABRT"}.get(rc, f"signal {-rc}")
            logger.warning(f"Subprocess PID={child_pid} crashed with {sig_name} (rc={rc})")

        # Check if result files exist — even FAILED/crashed scenarios may have partial results
        result_file = os.path.join(scenario_dir, 'result.json')
        has_results = os.path.exists(result_file) and os.path.getsize(result_file) > 0
        return has_results, scenario_dir, child_pid

    def execute_population(self, individuals):
        max_workers = self.ctn_manager.capacity
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures = []

        def run_one(ind_index, ind):
            try:
                ctn_cfg = self.ctn_manager.acquire(timeout=None)
                scenario_dir = os.path.join(self.result_folder, f"{ind.id}")

                run_status, scenario_dir, child_pid = self.execute_instance(
                    venv_dir=GlobalConfig.ads_venv_dir,
                    scenario_execute_script=GlobalConfig.scenario_executor_script,
                    scenario_entry_point=self.SCENARIO_ENTRY,
                    scenario_config=ind.scenario,
                    ctn_config=ctn_cfg,
                    scenario_dir=scenario_dir,
                    manager_name=self.MANAGER_NAME,
                    max_sim_time=GlobalConfig.max_sim_time,
                    debug=GlobalConfig.debug,
                    pytree_debug=GlobalConfig.pytree_debug,
                    open_vis=GlobalConfig.open_vis,
                )

                return {
                    "index": ind_index,
                    "status": run_status,
                    "scenario_dir": scenario_dir,
                    "gpu": ctn_cfg.gpu,
                    "reason": "normal",
                }
            except Exception as e:
                logger.exception(f"[Thread Worker] Exception {ind_index}: {e}")
                return {
                    "index": ind_index,
                    "status": False,
                    "scenario_dir": None,
                    "gpu": None,
                    "reason": "exception",
                    "error": str(e),
                }
            finally:
                self.ctn_manager.release(ctn_cfg)

        for ind_index, ind in enumerate(individuals):
            futures.append(executor.submit(run_one, ind_index, ind))

        results = []
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

        executor.shutdown(wait=True)
        self.cleanup_all_subprocesses()
        return results
