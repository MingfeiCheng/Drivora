import os
import ujson
import imageio
from pathlib import Path

base = "eval/cvprCKPT_simlingo_base/routes_validation/1/"
inspect_infractions = ["yield_emergency_vehicle_infractions","collisions_pedestrian","collisions_vehicle", "collisions_layout", "red_light", "stop_infraction", "scenario_timeouts", "outside_route_lanes", "vehicle_blocked", "route_dev"]

for result in os.listdir(base + "res"):
    res_idx = result.split("_")[0]
    with open(base+"res/"+result) as f:
        res = ujson.load(f)

    if res["_checkpoint"]["progress"][0] < res["_checkpoint"]["progress"][1]:
        continue

    for infraction_name in inspect_infractions:
        for i, infraction in enumerate(res["_checkpoint"]["records"][0]["infractions"][infraction_name]):
            infraction_frame = int(infraction.split("at Frame: ")[-1])

            infraction_frames = []
            frames = os.listdir(base+"viz/"+res_idx+"/images/")
            for frame in range(infraction_frame-50, infraction_frame+51):
                if f"{frame:04}.png" in frames:
                    infraction_frames.append(f"{base}viz/{res_idx}/images/{frame:04}.png")
                elif f"{frame}.png" in frames:
                    infraction_frames.append(f"{base}viz/{res_idx}/images/{frame}.png")

            os.makedirs(base+f"infractions/{infraction_name}", exist_ok=True)
            images = []
            for filename in infraction_frames:
                images.append(imageio.imread(filename))
            imageio.mimsave(f'{base}infractions/{infraction_name}/{res_idx}_{i}.gif', images, fps=2, loop=0)