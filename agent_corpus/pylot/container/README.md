## 🚗 Drivora-Carla-Pylot

This guide explains how to set up and use the **Pylot** ADS agent within the Drivora framework using the CARLA simulator.

> ⚠️ **Note:**  
Pylot is a modular, system-level ADS with complex dependencies. To simplify integration, Drivora uses a **pre-built container** where the Pylot runtime is already installed.  
The core implementation of Pylot resides in the `pylot/agent` directory.

---

### 🛠️ Step 1: Build the Base Environment

You have two options for preparing the base environment:

#### ✅ Option 1: Local Build

Run the following script to build the base image locally:

```bash
bash build_base.sh
```

This installs all required system dependencies and sets up the runtime environment from scratch.

#### 🌐 Option 2: Pull from Docker Hub

Alternatively, you can pull a pre-built base image:

```bash
docker pull mingfeicheng/drivora:pylot_base_0.9.10.1
```

---

### 🧱 Step 2: Build the Custom Environment

Depending on your choice in Step 1:

#### ▶️ If you **built locally**, simply run:

```bash
bash build_custom.sh
```

#### ▶️ If you **pulled the base image**, update the base image reference in `Dockerfile.custom`:

```dockerfile
FROM mingfeicheng/drivora:pylot_base_0.9.10.1
```

Then build the custom image:

```bash
bash build_custom.sh
```

You can verify that the image was created successfully by running:

```bash
docker images
```

You should see something like:

```
drivora/pylot    0.9.10.1
```

---

### 🐞 Step 3: Debug and Iterate

Unlike other end-to-end ADSs in Drivora, **modifying Pylot requires updating the code inside the container**.

In the Docker image, Pylot is installed at:

```
/home/erdos/workspace/pylot
```

To apply your changes:

1. Edit the relevant source files in your local `pylot/agent` directory.
2. In `Dockerfile.custom`, add or update the following line to overwrite the in-container directory:

```dockerfile
COPY ./agent /home/erdos/workspace/pylot
```

3. If new dependencies are needed, add them to `install_custom.sh`.
4. Rebuild the custom image:

```bash
bash build_custom.sh
```

> 💡 Drivora’s two-stage (base + custom) image design allows for fast iteration.  
> Heavy dependencies are built once in the base image, while source code changes are handled in the custom layer.

---

### ❗ Troubleshooting

Due to Pylot's complex dependency stack, you may occasionally encounter runtime or compatibility issues during modification.  
Some dependencies might need to be rebuilt or reinstalled inside the container depending on your changes.

> 🛠️ If you run into problems, feel free to open an issue or reach out for support.
