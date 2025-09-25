import threading
import argparse
import os
import sys
import copy
import carla
import zmq
import time
import traceback
import msgpack
import msgpack_numpy as m # pip install msgpack-numpy
m.patch() # enables numpy <-> msgpack compatibility

from loguru import logger

from atomic.data_provider import DataProvider
from atomic.timer import GameTime
from atomic.agent_wrapper import AgentWrapper
from atomic.navigation.local_planner import RoadOption

from agent import AgentSystem # TODO: support more flexible operations

REP_PORT = 12667
AGENT_CONFIG = "configs/challenge_map.conf"

class AgentNode:
    """
    Usage pipeline:
    
    1. Connect to the CARLA server
    2. Load the agent & bid the agent to an actor id in the CARLA world & setup sensors
    3. wait for start command with a global plan
    
    """
    
    def __init__(self, host, port):
        
        # carla connection parameters
        self.host = host
        self.port = port

        # wrapped agent
        self.actor_id = None
        self.agent = None
        self.agent_running = False
        # flag for creating the agent
        self._create_ads = False  # whether to create the agent or not
        self._in_setup_ad = False
        # plans
        self.plan_gps_carla = None
        self.plan_world_coord_carla = None
        
        # internal parameters
        self._timestamp_last_run = 0
        self._step = 0
        
        # connection carla status
        self.connection_status = "disconnected"
        self.setup_status = "not_ready"
        
        # receive channel for the agent
        self._ctx = zmq.Context()
        self._rep_socket = self._ctx.socket(zmq.REP)
        self._rep_socket.bind(f"tcp://0.0.0.0:{REP_PORT}")
        time.sleep(0.5)
        
        # some threads
        # Start command listener thread
        self._receive_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        
        # lock & lock parameters
        self._lock = threading.Lock()
        self._curr_return = None
    
    # receive command thread
    def _listen_for_commands(self):
        while True:
            try:
                raw_msg = self._rep_socket.recv()
                msg = msgpack.unpackb(raw_msg, raw=False, object_hook=m.decode)
                logger.info(f"[AGENT] Received command: {msg}")

                cmd = msg.get("cmd")
                args = msg.get("args", {})
                response = None

                # Dispatch control commands
                if cmd == "start":
                    self._command_start(args.get("plan_gps"), args.get("plan_world_coord"))
                    response = {"status": "ok", "message": "Agent starting."}
                    
                elif cmd == "stop":
                    self._command_stop()
                    response = {"status": "ok", "message": "Agent is stopping."}
                    
                elif cmd == "ping":
                    response = {"status": "ok", "message": "Agent alive."}

                elif cmd == "ping_carla":
                    if self.connection_status == "connected":
                        response = {
                            "status": "ok",
                            "message": f"Connected to Carla server at {self.host}:{self.port}."
                        }
                    else:
                        response = {"status": "error", "message": "Not connected to Carla server."}

                elif cmd == "setup":
                    actor_id = args.get("actor_id")
                    if actor_id is not None:
                        try:
                            self._command_setup(actor_id)
                            response = {"status": "ok", "message": f"Actor {actor_id} bid."}
                        except Exception as e:
                            logger.error(f"Failed to setup agent: {e}")
                            logger.error(traceback.format_exc())
                            response = {"status": "error", "message": str(e)}
                    else:
                        response = {"status": "error", "message": "Actor ID not provided."}

                elif cmd == "tick":
                    try:
                        with self._lock:
                            # payload = msgpack.packb(self._curr_return, default=m.encode)
                            response = {"status": "ok", "message": self._curr_return}
                    except Exception as e:
                        logger.warning(f"[ZeroMQ] Failed to generate tick response: {e}")
                        response = {"status": "error", "message": str(e)}

                elif cmd == "get_status":
                    # NOTE: the carla connection will end the main process, so, we only consider setup status
                    response = {"status": "ok", "message": f"{self.setup_status}"}
                
                else:
                    response = {"status": "error", "message": f"Unknown command: {cmd}"}

                self._rep_socket.send(msgpack.packb(response, default=m.encode))

            except Exception as e:
                logger.exception("Exception while handling command")
                error_response = {"status": "error", "message": str(e)}
                self._rep_socket.send(msgpack.packb(error_response, default=m.encode))

    def _command_setup(self, actor_id):
        self.actor_id = actor_id
        self._create_ads = True  # set the flag to create the agent
        logger.info(f"Setting up agent with actor ID: {actor_id}")
    
    def _command_start(self, plan_gps, plan_world_coord):
        """
        Start the agent node:
        1. set the global plan
        2. start the agent with a running flag
        """
        # convert to the json format
        plan_gps_carla = []
        for i, plan_gps_item in enumerate(plan_gps):
            plan_gps_carla.append((
                {'lat': plan_gps_item['lat'], 'lon': plan_gps_item['lon'], 'z': plan_gps_item['z']}, 
                RoadOption[plan_gps_item['connection']]))
        
        plan_world_coord_carla = []
        for i, plan_world_coord_item in enumerate(plan_world_coord):
            plan_world_coord_carla.append((
                carla.Transform(
                    carla.Location(
                        x=plan_world_coord_item['x'],
                        y=plan_world_coord_item['y'],
                        z=plan_world_coord_item['z']
                    ),
                    carla.Rotation(
                        pitch=plan_world_coord_item['pitch'],
                        yaw=plan_world_coord_item['yaw'],
                        roll=plan_world_coord_item['roll']
                    )
                ),
                RoadOption[plan_world_coord_item['road_option']]
            ))
        
        self.plan_gps_carla = plan_gps_carla
        self.plan_world_coord_carla = plan_world_coord_carla
        self.agent_running = True
        
        logger.info("Set agent routes successfully.")
    
    def _command_stop(self):
        """
        Stop the agent node:
        1. stop the agent with a running flag
        2. destroy the agent
        """
        logger.info("Stopping the agent...")
        
        with self._lock:
            self.agent_running = False
        
        logger.info("Agent node stopped successfully.")
        
    ####### Main Thread Methods #######
    def connect_to_carla(self, max_retries: int = 5, initial_delay: float = 1.0):
        """
        Connect to the CARLA server and initialize the DataProvider.
        Implements connection retries with exponential backoff.
        
        Args:
            max_retries (int): Number of connection attempts before giving up.
            initial_delay (float): Initial delay in seconds before retrying (doubles each time).
        """
        self.connection_status = "disconnected"

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to CARLA server at {self.host}:{self.port} (Attempt {attempt + 1}/{max_retries})...")
                client = carla.Client(self.host, self.port)
                client.set_timeout(20.0)

                world = client.get_world()
                if world is None:
                    raise RuntimeError("Could not retrieve the world from the CARLA server.")

                DataProvider.setup(client, world)
                self.connection_status = "connected"
                logger.info(f"Successfully connected to CARLA server at {self.host}:{self.port}.")
                return  # Exit the method if successful

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    backoff_time = initial_delay * (2 ** attempt)
                    logger.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                else:
                    logger.error(f"Failed to connect to CARLA server after {max_retries} attempts.")
                    raise RuntimeError(f"CARLA connection failed after {max_retries} retries: {e}")
    
    def _main_setup(self, actor_id):
        """
        Setup the agent node:
        1. create the agent
        2. create agent wrapper & register sensors
        3. reset the timer
        Args:
            actor_id (_type_): _description_
        """
        self.setup_status = "not_ready"
        
        # 1. create the agent & bid the actor id
        _agent = AgentSystem()
        _agent.setup(AGENT_CONFIG)
        _agent.bid_actor(actor_id)
        
        # 2. create agent wrapper
        logger.info("Setting up agent wrapper...")
        self.agent = AgentWrapper(_agent)
        logger.info("Agent wrapper created successfully. Setting up sensors...")
        self.agent.setup_sensors()
                
        # 3. reset the timer
        GameTime.restart()
        
        self.setup_status = "ready"
        
    def _main_stop(self):
        """
        Stop the agent node:
        1. stop the agent with a running flag
        2. destroy the agent
        """        
        self._curr_return = None  # reset the current return
        self.agent_running = False  # stop the agent
        
        # destroy the agent
        logger.info("Destroying the agent...")
        if self.agent:
            # NOTE: we only clean the sensors of this agent!
            self.agent.cleanup()
            self.agent = None
            self.actor_id = None
            self._create_ads = False  # reset the flag
            self.plan_gps_carla = None  # reset the global plan
            self.plan_world_coord_carla = None  # reset the global plan
            
            
        # reset the timer
        GameTime.restart()
      
    def _main_run(self):
        #  start the running loop
        # CAN Enter
        logger.info(f"Agent running thread started. {self.agent_running}")
        # start the agent
        self._timestamp_last_run = 0
        self._step = 0
        
        while True:
            with self._lock:
                if not self.agent_running:
                    logger.info("Agent is not running. Exiting run loop.")
                    break
                
            # tick the scenario        
            timestamp = None
            world = DataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                _agent_return = self._main_tick(timestamp)
                if _agent_return is not None:
                    with self._lock:
                        self._curr_return = copy.deepcopy(_agent_return)
               
            # time.sleep(0.1) # in case of overloading the server
            
    def _main_tick(self, timestamp):
        # default agent output
        _agent_output = None
        if self.agent_running and self._timestamp_last_run < timestamp.elapsed_seconds:
            self._timestamp_last_run = timestamp.elapsed_seconds
            self._step += 1
            
            # 1. tick timer
            GameTime.on_carla_tick(timestamp)
            
            # 2. run the agent
            try:
                _agent_control, _agent_log = self.agent()
                _agent_output = {
                    'status': 'running',
                    'local_step': self._step,
                    'control': {
                        'steer': _agent_control.steer,
                        'throttle': _agent_control.throttle,
                        'brake': _agent_control.brake,
                        'hand_brake': _agent_control.hand_brake,
                        'manual_gear_shift': _agent_control.manual_gear_shift
                    },
                    'logs': _agent_log
                }
                logger.info(f"Agent control: {_agent_control}, Log: {_agent_log}")
            except Exception as e:
                logger.error(f"Error running agent: {e}")
                logger.error(traceback.format_exc())
                _agent_output = {
                    'status': 'error',
                    'local_step': self._step,
                    'control': {
                        'steer': 0.0,
                        'throttle': 0.0,
                        'brake': 0.0,
                        'hand_brake': False,
                        'manual_gear_shift': False
                    },
                    'logs': {
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                }
                            
            # NOTE: Here we not tick the scenario, this operation is done in the host runner
        
        return _agent_output

    def main_thread(self):
        self._running_main = True
        self._receive_thread.start()
        logger.info("Agent node is now hanging and waiting for commands.")
        while self._running_main:
            
            if self._create_ads and self.agent is None and not self._in_setup_ad:
                self._in_setup_ad = True
                self._main_setup(self.actor_id)
                self._in_setup_ad = False
                
            # if the agent is running, we run the agent
            if self.agent and self.agent_running:
                # 1. set the global plan
                self.agent.setup_route(
                    self.plan_gps_carla, 
                    self.plan_world_coord_carla
                )
                self._main_run()                                
                self._main_stop()  # stop the agent after running
                
            time.sleep(0.1)
            
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--host", type=str, default="localhost") # carla host
    arg_parser.add_argument("--port", type=int, default=2000) # carla port
    args = arg_parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    level = 'DEBUG'
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])
    logger_file = os.path.join(log_dir, 'run.log')

    logger.add(
        logger_file,
        level=level,
        mode="a",  # Append mode
        rotation="10 MB",  # Create a new log file when size > 10MB
        retention="5 days",  # Delete logs older than 5 days
        compression="zip",  # Optional: Compress old logs
    )
    

    logger.info("Starting Agent Node...")
    agent_node = AgentNode(args.host, args.port)
    
    # Steps:
    # 1. connect to CARLA server
    try:
        agent_node.connect_to_carla()
    except RuntimeError as e:
        agent_node.stop()
        logger.error(f"Failed to connect to CARLA server: {e}")
        sys.exit(1)
    
    # 2. running the node to wait for commands
    logger.info("Agent Node is running and waiting for commands...")
    agent_node.main_thread()    
    logger.info("Agent Node has shut down.")