import os
import importlib

def discover_modules(root_dir, exclude_dirs=None, package_name=None):
    """
    Discover and import all Python modules under the specified root directory.

    Args:
        root_dir (str): The root directory to search.
        exclude_dirs (list, optional): List of directory names to exclude (e.g., ['tests', '__pycache__']).
        package_name (str, optional): The root package name to prepend to module imports.
    """
    if exclude_dirs is None:
        exclude_dirs = ["tests", "__pycache__"]

    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # if "__init__.py" not in files:
        #     continue  # Skip this folder and its subdirectories

        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Build the module name from the relative path
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

                if package_name:
                    full_module_name = f"{package_name}.{module_name}"
                else:
                    full_module_name = module_name

                # Import the module dynamically
                try:
                    importlib.import_module(full_module_name)
                except Exception as e:
                    print(f"Failed to import {full_module_name}: {e}")