# Gaussian Splatting-based Localization

A visual localization pipeline that estimates the 6DoF camera pose of a target RGB image using 3D Gaussian Splatting (3DGS) as the underlying scene representation. 

This method follows a two-stage design:
1. **Initial Pose Estimation:** Estimates an initial camera pose from a sparse Structure-from-Motion (SfM) point cloud using Hierarchical Localization (HLOC) components for global retrieval (NetVLAD) and local feature extraction (SuperPoint), followed by Perspective-n-Point (PnP) with RANSAC.
2. **Pose Refinement:** Iteratively refines the pose through analysis-by-synthesis optimization by minimizing the photometric discrepancy between the query image and a view rendered from the explicit 3DGS map.

The detailed report on the project is in the [report](report.pdf).

## Installation

The following guide outlines the necessary steps to set up the environment, including the original Gaussian Splatting submodule and its CUDA dependencies. 

```bash
# 1. Clone this repository
git clone https://github.com/Vladick102/GS-based-Localization.git
# or via SSH: 
git clone git@github.com:Vladick102/GS-based-Localization.git

# 2. Clone the 3D Gaussian Splatting repository recursively
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# 3. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install the required Python dependencies
pip install -r requirements.txt

# 5. Install the simple-knn module
pip install --no-build-isolation ./gaussian-splatting/submodules/simple-knn

# 6. Patch the CUDA Rasterizer for compatibility
# Add missing includes to the rasterizer_impl.h file:
# Open the file using nano (or your preferred editor):
nano gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h

# Add the following two lines near the top of the file:
# #include <cstddef>
# #include <cstdint>

# 7. Install the diff-gaussian-rasterization module
pip install --no-build-isolation ./gaussian-splatting/submodules/diff-gaussian-rasterization

# 8. Clone and link SuperGluePretrainedNetwork
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Symlink the cloned SuperGlue network directly to site-packages so it can be imported globally
ln -s $(pwd)/SuperGluePretrainedNetwork "$SITE_PACKAGES/SuperGluePretrainedNetwork"
```

## Usage

You can use the `main.py` entry point to localize a query image. The system requires both the SfM point cloud (used for the initial PnP pose) and the trained 3DGS model (used for photometric refinement).

### Example Data

You can download the `bonsai` (SfM model and test images) and `bonsai-gs` (pretrained 3DGS model) folders, from the Mip-NeRF 360 dataset, used to test the pipeline [here](https://drive.google.com/drive/folders/1DBh4yzgd91AjCBMx26FNR3i0XoD8WA7G?usp=sharing).

### Required Input Parameters

* `--scene`: The directory containing the SfM (Structure-from-Motion) sparse reconstruction assets (e.g., COLMAP output) or the `scene.json`/`cameras.json`. This provides the 3D reference points for the initial pose guess.
* `--gs-model`: The directory containing the trained 3D Gaussian Splatting model.
* `--gs-repo`: The path to the cloned `gaussian-splatting` repository (used for rendering).
* `--query`: The path to the target query RGB image you want to localize.
* `--output`: The destination path where the predicted 6DoF camera pose will be saved as a JSON file.

### Example Command

Assuming you are running from the root of the project:

```bash
python3 GS-based-Localization/main.py localize \
    --scene bonsai/ \
    --gs-model bonsai-gs/ \
    --gs-repo gaussian-splatting/ \
    --query image.png \
    --output result.json
```
