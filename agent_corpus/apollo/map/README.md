# CARLA → Apollo HD maps

Converts CARLA towns (OpenDRIVE `.xodr`) into complete Apollo HD maps and
installs them into the Apollo tree, so Apollo's localization/routing/planning
can run on CARLA layouts.

## Pipeline (see `convert_carla_map.sh`)
1. **Sanitize** the CARLA `geoReference`: drop `+geoidgrids=egm96_15.gtx`
   / `+vunits=m` (CARLA towns are flat; pyproj lacks the egm96 grid).
2. **`imap_box`** (pip converter by daohu527, an Apollo contributor):
   `imap -f -i <xodr> -o base_map.txt` → emits `base_map.txt` + `base_map.bin`.
3. **Apollo's own tools** (run in a disposable Apollo container, not touching any
   running experiment): `topo_creator` → `routing_map.{txt,bin}`;
   `sim_map_generator` → `sim_map.{txt,bin}`.

Only `imap_box` + Apollo's own tools are used — nothing external except the
Apollo source tree.

## Convert another town
```bash
bash convert_carla_map.sh Town04 /home/$USER/carlaCache/Carla/Maps/OpenDrive/Town04.xodr
```

## Town01 (done & validated)
Installed at `<apollo>/modules/map/data/Town01/` and copied here.

| file | content |
|---|---|
| base_map.bin    | 300 lanes, 12 junctions, 33 signals, 97 overlaps |
| routing_map.bin | 124 nodes, 160 edges (routing topology) |
| sim_map.bin     | 300 lanes, 300 roads |

- **Projection:** `+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m` —
  base_map coords are UTM zone 31 metres. The proxy's GNSS→map projection in
  `../transform.py` MUST target this same UTM frame (see calibration in
  `../config/proxy_config.json`).
- Validated offline: all three files parse as Apollo protos and routing topo is
  non-empty (`debug/.../map_validation.json`).

## Using the map
Select per-run via `--map_dir=/apollo/modules/map/data/Town01` (or `map_name:
"Town01"` in `proxy_config.json`). **Do not edit `global_flagfile.txt`** — it is
shared with other running Apollo experiments.
