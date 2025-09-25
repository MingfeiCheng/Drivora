import os
import sys
import importlib
import traceback
from loguru import logger

FOLDERS = [
    "fuzzer",
    "scenario_runner",
]

def discover_modules(root_dir):
    # Ensure root_dir is in sys.path so modules like fuzzer.xxx can be imported
    abs_root = os.path.abspath(root_dir)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)

    for sub_folder in FOLDERS:
        sub_dir = os.path.join(root_dir, sub_folder)
        if not os.path.isdir(sub_dir):
            logger.warning(f"Subdirectory {sub_dir} does not exist. Skipping.")
            continue

        logger.info(f"Discovering modules in: {sub_dir}")
        for root, dirs, files in os.walk(sub_dir):
            # Optionally skip hidden or cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    # Relative to root_dir, so full package path works
                    abs_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_file_path, root_dir)
                    module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

                    try:
                        importlib.import_module(module_name)
                        # logger.info(f"✅ Imported: {module_name}")
                    except Exception as e:
                        traceback.print_exc()
                        logger.error(f"❌ Failed to import {module_name}: {e}")
