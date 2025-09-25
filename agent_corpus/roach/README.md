## 🚗 Drivora-Carla-Roach

This guide explains how to set up and use the **Roach** ADS agent within the Drivora framework using the CARLA simulator.

---

### 📦 Step 1: Prerequisites

#### 🔗 Pretrained Weights

Download the pretrained model weights from the link below:

[📥 Roach Pretrained Weights (Google Drive)](https://drive.google.com/file/d/1dg_nB8OvB9H-wpcSlpUhXNgrXW8_pPIQ/view?usp=drive_link)

After downloading, place the weight file in the following directory:

```
roach/log/
```

> 💡 Ensure the filename matches what the Roach agent expects in its loading script.

---

### 🛠️ Step 2: Build the Base Environment

You have two options to prepare the base environment:

#### ✅ Option 1: Local Build

All dependencies required by Roach are predefined. Simply run:

```bash
bash build_base.sh
```

This will build a local base image with the necessary runtime environment.

#### 🌐 Option 2: Pull from Docker Hub

We also provide a pre-built base image on Docker Hub:

```bash
docker pull mingfeicheng/drivora:roach_base_0.9.10.1
```

---

### 🧱 Step 3: Build the Custom Environment

Depending on how you completed Step 2:

#### ▶️ If you **built locally**, simply run:

```bash
bash build_custom.sh
```

#### ▶️ If you **pulled the base image**, make sure to update the base image reference in `Dockerfile.custom`:

```dockerfile
FROM mingfeicheng/drivora:roach_base_0.9.10.1
```

Then run:

```bash
bash build_custom.sh
```

After the build, verify your image is created:

```bash
docker images
```

You should see an image like:

```
drivora/roach    0.9.10.1
```

---

### 🐞 Step 4: Debug and Iterate

To test and iterate on your custom Roach agent:

1. Modify any source files (e.g., `agent.py`, model loading logic, etc.)
2. Add any additional dependencies to `install_custom.sh`
3. Rebuild only the custom image:

```bash
bash build_custom.sh
```

> ✅ Drivora separates the base and custom layers to reduce rebuild time.  
> Heavy dependencies should be placed in the base image, while lightweight changes can be handled through the custom image.

---

Let me know if you want to include a **run script** or evaluation example as a follow-up!
