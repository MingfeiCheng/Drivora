from loguru import logger

class DataProvider:
    
    _client = None
    _world = None
    _map = None
    _sync_flag = False
    
    @staticmethod
    def setup(client, world):
        DataProvider._client = client
        DataProvider._world = world
        DataProvider._map = world.get_map()
        DataProvider._sync_flag = world.get_settings().synchronous_mode
        logger.info("DataProvider initialized with client and world. Synchronous mode: {}", DataProvider._sync_flag)

    @staticmethod
    def get_client():
        if DataProvider._client is None:
            raise RuntimeError("DataProvider not initialized. Call setup() first.")
        return DataProvider._client
    
    @staticmethod
    def get_world():
        if DataProvider._world is None:
            raise RuntimeError("DataProvider not initialized. Call setup() first.")
        return DataProvider._world
    
    @staticmethod
    def get_map():
        if DataProvider._map is None:
            raise RuntimeError("DataProvider not initialized. Call setup() first.")
        return DataProvider._map
    
    @staticmethod
    def is_sync_mode():
        """
        @return true if syncronuous mode is used
        """
        return DataProvider._sync_flag