import time
import carla
import docker
import subprocess

from numpy import random
from loguru import logger
from typing import Optional

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version # Python 2 fallback
    
try:
    # requires Python 3.8+
    from importlib.metadata import metadata
    def get_carla_version():
        return Version(metadata("carla")["Version"])
except ModuleNotFoundError:
    # backport checking for older Python versions; module is deprecated
    import pkg_resources
    def get_carla_version():
        return Version(pkg_resources.get_distribution("carla").version)

OLD_CARLA_VERSION = Version("0.9.12")

"""
Current implementation only supports one client
"""

class CtnSimOperator:
    # here, it should provide the basic operations for the carla, yes
    def __init__(
        self, 
        idx: int,
        container_name,
        gpu, 
        random_seed=0,
        docker_image='carlasim/carla:0.9.15', 
        fps=20,
        is_sync_mode=True
    ):
        self.idx = idx
        self.fps = fps
        self.container_name = container_name
        self.port = int(2000)
        self.tm_port = self.port + 2 * self.idx + 1
        self.gpu = gpu
        self.timeout = 20.0
        self.docker_image = docker_image
        self.is_sync_mode = is_sync_mode
        self.random_seed = random_seed
        
        # carla info if needed
        self.client = None
        self.world = None
        self.map = None
        
        self.carla_version = get_carla_version()
        logger.info(f"Detected Carla version: {self.carla_version}")
        
        # some internal parameters for global managerment
        self._carla_actor_pool = {}      # type: dict[int, carla.Actor]
        self._rng = random.RandomState(self.random_seed)

    @property
    def host(self):
        while not self.is_running:
            self.start()
        ctn = docker.from_env().containers.get(self.container_name)
        return ctn.attrs['NetworkSettings']['IPAddress']
    
    @property
    def is_running(self) -> bool:
        """
        Checks if the container is running

        :returns: True if running, False otherwise
        :rtype: bool
        """
        try:
            return docker.from_env().containers.get(self.container_name).status == 'running'
        except:
            return False

    @property
    def is_connected(self) -> bool:
        """
        Checks if the Carla server is connected

        :returns: True if connected, False otherwise
        :rtype: bool
        """
        if self.client is None:
            return False
        
        try:
            self.client.get_world()
            return True
        except:
            return False

    def _start_operation(self, wait_time=5.0, max_wait=60.0):
        # before 0.9.12
        if self.carla_version < OLD_CARLA_VERSION:
            cmd = f"docker run --privileged --name=\"{self.container_name}\" -d --rm " \
                f"--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={self.gpu} " \
                f"--gpus 'device={self.gpu}' " \
                f"{self.docker_image} " \
                f"/bin/bash -c 'SDL_VIDEODRIVER=offscreen CUDA_DEVICE_ORDER=PCI_BUS_ID " \
                f"CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh " \
                f"-nosound -windowed -opengl " \
                f"-carla-rpc-port={self.port}'"
        else:
            cmd = f"docker run --privileged --name=\"{self.container_name}\" -d --rm " \
                f"--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={self.gpu} " \
                f"--gpus 'device={self.gpu}' -v /tmp/.X11-unix:/tmp/.X11-unix:rw " \
                f"{self.docker_image} " \
                f"/bin/bash ./CarlaUE4.sh -RenderOffScreen"
                
        
        logger.info(cmd)
        process = subprocess.run(cmd, shell=True)
        
        # Wait for the container to be up
        # logger.info("Waiting for container to be in 'running' state...")
        start_time = time.time()
        while True:
            check_cmd = f"docker ps -q -f name=^{self.container_name}$"
            result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.stdout.strip():
                logger.info(f"Container '{self.container_name}' is now running.")
                break

            if time.time() - start_time > max_wait:
                logger.error(f"Timeout: Container '{self.container_name}' did not start within {max_wait} seconds.")
                raise TimeoutError(f"Container '{self.container_name}' did not start in time.")

            time.sleep(1.0)  # poll every 1 second

        time.sleep(wait_time)
        
    def start(self):
        max_retries = 20
        wait_time = 5.0

        if not self.is_running:
            self._start_operation(wait_time)

        if self.is_connected:
            # will load world to destroy all actors
            self.world = self.client.load_world("Town01")  # TODO: configurable map
            return

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Trying to connect to Carla at {self.host}:{self.port} (attempt {attempt}/{max_retries})")

                self.client = carla.Client(self.host, int(self.port))
                self.client.set_timeout(20.0 + wait_time)
                self.world = self.client.load_world("Town01")  # TODO: configurable map

                logger.info(f"Successfully connected to Carla at {self.host}:{self.port}")
                return  # success
            except Exception:
                logger.exception(f"Failed to connect to Carla (attempt {attempt}/{max_retries})")
                self.stop()
                self._start_operation(wait_time)
                wait_time *= 1.2  # exponential backoff

        # if loop finishes without return
        logger.error("Exceeded maximum retries (%d). Could not connect to Carla.", max_retries)
        raise SystemExit(-1)

    def stop(self):
        cmd = f"docker stop {self.container_name}"
        process = subprocess.run(cmd, shell=True)
        logger.info('Stop container: {}', self.container_name)
        time.sleep(1.0)
    
    def remove(self):
        logger.warning(f"Removing existing container '{self.container_name}' from environment...")
        cmd = f"docker rm -f {self.container_name}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    ########### CARLA related operations ###########
    def cleanup(self):
        """
        Cleanup and remove all entries from all dictionaries
        """
        DestroyActor = carla.command.DestroyActor  # pylint: disable=invalid-name
        batch = []

        for actor_id in self._carla_actor_pool.copy():
            actor = self._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                batch.append(DestroyActor(actor))

        if self.client:
            try:
                self.client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e
                
        self._map = None
        self._world = None
        self._sync_flag = False
        self._ego_vehicle_route = None
        self._all_actors = None
        self._carla_actor_pool = {}
        self._client = None
        self._spawn_points = None
        self._spawn_index = 0
        self._rng = random.RandomState(self.random_seed)
        self._grp = None
        self._runtime_init_flag = False
        self._latest_scenario = ""
        
    def get_world(self) -> Optional[carla.World]:
        if self.world is None:
            try:
                self.world = self.client.get_world()
            except Exception as e:
                logger.error(f"Failed to get Carla world: {e}, trying restart...")
                try:
                    self.start()
                    self.world = self.client.get_world()
                except Exception as e2:
                    logger.error(f"Restart failed: {e2}")
                    return None
                
        return self.world
    
    def get_map(self) -> Optional[carla.Map]:
        if self.map is None:
            world = self.get_world()
            if world is not None:
                try:
                    self.map = world.get_map()
                except Exception as e:
                    logger.error(f"Failed to get Carla map: {e}")
                    return None
            
        return self.map
    
    ####### Reusable tools for scenario manager ########
    def request_new_actor(
        self,
        model, 
        spawn_point, 
        rolename='scenario', 
        autopilot=False,
        color=None, 
        actor_category="car",
        attribute_filter=None, 
        tick=True
    ) -> Optional[carla.Actor]:
        """
        This method tries to create a new actor, returning it if successful (None otherwise).
        """
        
        blueprint = self.create_blueprint(model, rolename, color, actor_category, attribute_filter)

        # For non prop models, slightly lift the actor to avoid collisions with the ground
        z_offset = 0.2 if 'prop' not in model else 0

        # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
        _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
        _spawn_point.location.x = spawn_point.location.x
        _spawn_point.location.y = spawn_point.location.y
        _spawn_point.location.z = spawn_point.location.z + z_offset
        actor = self.world.try_spawn_actor(blueprint, _spawn_point)

        if actor is None:
            logger.warning("WARNING: Cannot spawn actor {} at position {}".format(model, spawn_point.location))
            return None

        # De/activate the autopilot of the actor if it belongs to vehicle
        if autopilot:
            if isinstance(actor, carla.Vehicle):
                actor.set_autopilot(autopilot, self.carla_tm_port)
            else:
                logger.warning("WARNING: Tried to set the autopilot of a non vehicle actor")

        # Wait for the actor to be spawned properly before we do anything
        if not tick:
            pass
        elif self.is_sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if actor is None:
            return None

        self._carla_actor_pool[actor.id] = actor
        return actor
    
    def create_blueprint(
        self,
        model, 
        rolename='scenario', 
        color=None, 
        actor_category="car",
        attribute_filter=None
    ):
        # type: (str, str, carla.Color | None, str, dict | None) -> carla.ActorBlueprint
        """
        Function to setup the blueprint of an actor given its model and other relevant parameters
        """

        def check_attribute_value(blueprint, name, value):
            """
            Checks if the blueprint has that attribute with that value
            """
            if not blueprint.has_attribute(name):
                return False

            attribute_type = blueprint.get_attribute(name).type
            if attribute_type == carla.ActorAttributeType.Bool:
                return blueprint.get_attribute(name).as_bool() == value
            elif attribute_type == carla.ActorAttributeType.Int:
                return blueprint.get_attribute(name).as_int() == value
            elif attribute_type == carla.ActorAttributeType.Float:
                return blueprint.get_attribute(name).as_float() == value
            elif attribute_type == carla.ActorAttributeType.String:
                return blueprint.get_attribute(name).as_str() == value

            return False

        _actor_blueprint_categories = {
            'car': 'vehicle.tesla.model3',
            'vehicle': 'vehicle.tesla.model3',
            'van': 'vehicle.volkswagen.t2',
            'truck': 'vehicle.carlamotors.carlacola',
            'trailer': '',
            'semitrailer': '',
            'bus': 'vehicle.volkswagen.t2',
            'motorbike': 'vehicle.kawasaki.ninja',
            'bicycle': 'vehicle.diamondback.century',
            'train': '',
            'tram': '',
            'pedestrian': 'walker.pedestrian.0001',
            'misc': 'static.prop.streetbarrier'
        }

        # Set the model
        try:
            if "vehicle.lincoln.mkz2017" in model or "vehicle.lincoln.mkz_2017" in model:
                if self.carla_version < OLD_CARLA_VERSION:
                    model = "vehicle.lincoln.mkz2017"
                else:
                    model = "vehicle.lincoln.mkz_2017"
            
            blueprints = self.world.get_blueprint_library().filter(model)
            if attribute_filter is not None:
                for key, value in attribute_filter.items():
                    blueprints = [x for x in blueprints if check_attribute_value(x, key, value)]
            blueprint = self._rng.choice(blueprints)
        except ValueError:
            # The model is not part of the blueprint library. Let's take a default one for the given category
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories[actor_category]
            if new_model != '':
                bp_filter = new_model
            logger.warning("WARNING: Actor model {} not available. Using instead {}".format(model, new_model))
            blueprint = self._rng.choice(self.world.get_blueprint_library().filter(bp_filter))

        # Set the color
        if color:
            if not blueprint.has_attribute('color'):
                logger.warning(
                    "WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(
                        color, blueprint.id
                    )
                )
            else:
                default_color_rgba = blueprint.get_attribute('color').as_color()
                default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
                try:
                    blueprint.set_attribute('color', color)
                except ValueError:
                    # Color can't be set for this vehicle
                    logger.warning(
                        "WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(
                            color, blueprint.id, default_color
                        )
                    )
                    blueprint.set_attribute('color', default_color)
        else:
            if blueprint.has_attribute('color') and rolename != 'hero':
                color = self._rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

        # Make pedestrians mortal
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        # Set the rolename
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', rolename)

        return blueprint
    
    
    def remove_actor(self, actor: carla.Actor):
        """
        Remove the actor from the simulation and from the internal pool
        """
        actor_id = actor.id
        
        if actor_id in self._carla_actor_pool:
            was_destroyed = self._carla_actor_pool[actor_id].destroy()
            self._carla_actor_pool[actor_id] = None  # type: ignore
            self._carla_actor_pool.pop(actor_id)
            return was_destroyed
        # logger.info("Trying to remove a non-existing actor id {}".format(actor_id))
        return None