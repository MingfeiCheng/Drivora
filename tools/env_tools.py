import os
import subprocess

def get_available_gpus():
    """Return a list of available GPU IDs"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Respect environment variable limitation
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        return [int(g.strip()) for g in visible_gpus if g.strip().isdigit()]
    try:
        # Use nvidia-smi to detect all GPUs
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            if line.startswith("GPU "):
                gpu_id = int(line.split()[1].strip(":"))
                gpus.append(gpu_id)
        return gpus
    except Exception as e:
        print(f"[WARN] Failed to detect GPUs with nvidia-smi: {e}")
        return []