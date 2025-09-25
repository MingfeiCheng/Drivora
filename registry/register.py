import fnmatch

from loguru import logger

class Registry:
    def __init__(self, name: str):
        """
        Initialize the registry.

        Args:
            name (str): The name of the registry.
        """
        self.name = name
        self._registry = {}  # Stores {name: module_path or class reference}

    def register(self, name=None):
        """
        Decorator to register a class with a name and path.

        Args:
            name (str, optional): The name to register the class under. Defaults to the class name.
        Returns:
            function: The decorator function.
        """
        def decorator(cls):
            register_name = name or cls.__name__
            if register_name in self._registry:
                raise KeyError(f"'{register_name}' is already registered in the '{self.name}' registry.")
            self._registry[register_name] = cls
            return cls
        return decorator

    def get(self, class_name):
        """
        Retrieve a registered class by its name.

        Args:
            class_name (str): The name of the class.

        Returns:
            type: The class object.

        Raises:
            KeyError: If the class is not registered.
        """
        # print(self._registry.keys())
        if class_name in self._registry:
            return self._registry[class_name]
        raise KeyError(f"Class '{class_name}' is not registered in the '{self.name}' registry.")

    def filter_actors(self, pattern):
        """
        Filter registered actors based on a pattern.

        Args:
            pattern (str): The pattern to match (e.g., "vehicle*").

        Returns:
            dict: A dictionary of actor names and their corresponding classes matching the pattern.
        """
        return {name: cls for name, cls in self._registry.items() if fnmatch.fnmatch(name, pattern)}

