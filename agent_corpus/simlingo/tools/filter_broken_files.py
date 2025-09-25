import glob
import json
import gzip
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

path = 'database/simlingo/dreamer'
file_ending = '.json.gz'

# Load all files in the directory
all_files = glob.glob(f'{path}/**/*{file_ending}', recursive=True)

# Print the number of files
print(len(all_files))

def check_file(file):
    try:
        with gzip.open(file, 'rt') as f:
            data = json.load(f)
        return None
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return file

# find the broken files using multiprocessing
broken_files = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(check_file, file): file for file in all_files}
    for future in tqdm.tqdm(as_completed(futures), total=len(all_files)):
        result = future.result()
        if result:
            broken_files.append(result)

# Print the number of broken files
print(len(broken_files))
breakpoint()