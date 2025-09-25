import os
import cv2
import gzip
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import defaultdict
from loguru import logger
from natsort import natsorted

def images_to_video(image_folder, output_path, fps=30, delete_folder=True):
    # Get list of image files (sorted naturally: 1.png, 2.png, ..., 10.png)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)  # sort filenames like humans

    if not image_files:
        logger.info("[!] No images found.")
        return

    # Read the first image to get size
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logger.info(f"[!] Skipping unreadable image: {img_file}")
            continue
        out.write(img)

    out.release()
    logger.info(f"[✓] Video saved to: {output_path}")
    
    if delete_folder:
        shutil.rmtree(image_folder)

def load_observation(scenario_dir):
    observation_file = os.path.join(scenario_dir, "observation.jsonl.gz")
    data = []
    with gzip.open(observation_file, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_runtime_result(scenario_dir):
    runtime_result_file = os.path.join(scenario_dir, "result.json")
    with open(runtime_result_file, "r") as f:
        data = json.load(f)
    return data


def visualize_trajectories(scenario_dir: str, downsample: int = 1):
    SCENARIO_DIR = scenario_dir
    SCENARIO_FILE = os.path.join(SCENARIO_DIR, "scenario.json")
    SAVE_PNG_CFG = os.path.join(SCENARIO_DIR, "vis_config.png")
    SAVE_PNG_TRAJ = os.path.join(SCENARIO_DIR, "vis_trajectories.png")

    POST_DOWNSAMPLE = 1

    # ========= Load JSON =========
    def load_json(path):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    scenario_data = load_json(SCENARIO_FILE)
    if scenario_data is None:
        logger.warning(f"[ERR] Missing scenario_config.json at {SCENARIO_FILE}")
        return

    obs_data_raw = load_observation(SCENARIO_DIR)  # maybe None

    # ========= Predefined trajectories =========
    pre_traj = defaultdict(list)
    for ego in scenario_data.get("ego_vehicles", []):
        ego_id = str(ego.get("id"))
        for wp in ego.get("route", []):
            pre_traj[ego_id].append((wp["x"], wp["y"]))

    for npc in scenario_data.get("npc_vehicles", []):
        npc_id = str(npc.get("id"))
        for wp in npc.get("route", []):
            pre_traj[npc_id].append((wp["x"], wp["y"]))

    # ========= Observed trajectories =========
    post_traj = defaultdict(list)
    if obs_data_raw:
        frames = obs_data_raw if isinstance(obs_data_raw, list) else []
        for idx, frame in enumerate(frames):
            if POST_DOWNSAMPLE > 1 and (idx % POST_DOWNSAMPLE != 0):
                continue
            for ego_id, ego_item in frame.get("egos", {}).items():
                loc = ego_item.get("location")
                post_traj[str(ego_id)].append((float(loc[0]), float(loc[1])))
            for npc_item in frame.get("other_actors", {}).get("vehicles", {}):
                npc_id = npc_item.get("config_id", "unknown")
                loc = npc_item.get("location")
                post_traj[str(npc_id)].append((float(loc[0]), float(loc[1])))

    # ========= Helper functions =========
    def plot_traj(ax, traj_xy, ego_id, phase="pre", style="-", color=None):
        if not traj_xy:
            return
        xs, ys = zip(*traj_xy)
        ax.plot(xs, ys, linestyle=style, label=f"{ego_id} ({phase})", color=color)
        # arrows
        for i in range(0, len(xs) - 1, max(1, len(xs)//10)):
            dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
            ax.arrow(xs[i], ys[i], dx, dy,
                     head_width=0.5, head_length=1.0,
                     fc=color, ec=color, alpha=0.7,
                     length_includes_head=True)

    def plot_endpoints(ax, traj_xy, ego_id, phase, color):
        if not traj_xy:
            return
        sx, sy = traj_xy[0]
        ex, ey = traj_xy[-1]
        ax.scatter([sx], [sy], marker="o", s=40, color=color, label=f"{ego_id} {phase} S")
        ax.scatter([ex], [ey], marker="x", s=40, color=color, label=f"{ego_id} {phase} E")

    def setup_ax(ax, title):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, linestyle=":")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best", ncol=2)

    # color map
    all_ids = sorted(set(pre_traj.keys()) | set(post_traj.keys()))
    color_map = cm.get_cmap("tab10", len(all_ids))
    id2color = {eid: color_map(i) for i, eid in enumerate(all_ids)}

    # ========= Figure 1: predefined config =========
    fig1, ax1 = plt.subplots(figsize=(8, 7), dpi=110)
    for ego_id in pre_traj:
        plot_traj(ax1, pre_traj[ego_id], ego_id, phase="pre", style="--", color=id2color[ego_id])
        plot_endpoints(ax1, pre_traj[ego_id], ego_id, "pre", id2color[ego_id])
    setup_ax(ax1, "Predefined Routes (scenario.json)")
    plt.tight_layout()
    plt.savefig(SAVE_PNG_CFG, bbox_inches="tight")
    logger.info(f"[OK] Saved config figure to: {SAVE_PNG_CFG}")
    plt.close(fig1)

    # ========= Figure 2: observed only =========
    fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=110)
    for ego_id in post_traj:
        plot_traj(ax2, post_traj[ego_id], ego_id, phase="post", style="-", color=id2color.get(ego_id, "black"))
        plot_endpoints(ax2, post_traj[ego_id], ego_id, "post", id2color.get(ego_id, "black"))
    setup_ax(ax2, "Observed Trajectories (from observation.jsonl.gz)")
    plt.tight_layout()
    plt.savefig(SAVE_PNG_TRAJ, bbox_inches="tight")
    logger.info(f"[OK] Saved trajectories figure to: {SAVE_PNG_TRAJ}")
    plt.close(fig2)


class DisplayInterface(object):

    def __init__(self):
        self._width = 1200
        self._height = 400
        self._surface = None

    def run_interface(self, input_data):
        rgb_front = cv2.cvtColor(input_data['recorder_rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb_front, (1200, 400))
        surface = np.zeros((400, 1200, 3), np.uint8)
        surface[:, :1200] = rgb

        if 'vis_text' in input_data:
            vis_text = input_data['vis_text']
            surface = cv2.putText(surface, vis_text['basic_info'], (20, 290), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255), 1)
            # surface = cv2.putText(surface, vis_text['plan'], (20, 710), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
            surface = cv2.putText(surface, vis_text['control'], (20, 330), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255),
                                  1)
            surface = cv2.putText(surface, vis_text['others'], (20, 370), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255),
                                  1)

        return surface
    
def get_recorder_sensors():
    return [
        {
            'type': 'sensor.camera.rgb',
            'x': - 10.0, 'y': 0.0, 'z': 2.0,
            'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
            'width': 1200, 'height': 400, 'fov': 100,
            'id': 'recorder_rgb_front'
        }
    ]
