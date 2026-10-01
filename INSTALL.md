# Installation

## 1. Environment Setup

Create the conda environment from the provided environment file:

```bash
conda env create -f planavid_env.yml
```

```
conda activate planavid
# OR
alias python='/path/to/your/conda/envs/planavid/bin/python3.8'
alias pip='/path/to/your/conda/envs/planavid/bin/pip3'
source /path/to/your/conda/bin/activate planavid
```

The environment was tested with CUDA 11.8 (in `/usr/local/`) and Python 3.8.

Some systems require additional OpenGL/EGL libraries for headless Habitat rendering:

```
apt-get update
apt-get install -y libegl1 libglvnd0 libopengl0 libgl1
```

Add the following environment variables to your shell configuration:

```
cat << 'EOF' >> ~/.bashrc
# Prioritize system drivers to avoid EGL symbol errors.
export SYS_LIB="/usr/lib/x86_64-linux-gnu"
export CUDA_LIB=$(ls -d /usr/local/cuda-*/lib64 2>/dev/null | head -n 1)
export LD_LIBRARY_PATH="$SYS_LIB:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Enable headless rendering for Habitat.
export HABITAT_SIM_FORCE_EGL_OFFSCREEN=1
export NVIDIA_DRIVER_CAPABILITIES=all
export CUDA_VISIBLE_DEVICES=0
EOF

source ~/.bashrc
```

## 2. Install Habitat-Sim and Habitat-Lab

```
conda install -y -c aihabitat -c conda-forge \
  habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7

git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
# TensorFlow 1.13 (tensorboard videos only) has no Python 3.8 wheel and is not needed
sed -i '/tensorflow==1.13.1/d' habitat_baselines/rl/requirements.txt

python -m pip install "msgpack==1.0.7"
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
cd ..
```

Then run the following commands:

```
apt-get update && apt-get install -y libopengl0
conda uninstall -y libglvnd libegl libglx --force

export LD_PRELOAD=$(find /usr/lib/x86_64-linux-gnu -name "libEGL_nvidia.so.0" | head -n 1)

pip uninstall -y numpy
conda install -y numpy=1.23.5 numpy-base=1.23.5
```

## 3. Download Checkpoints

Please organize the checkpoints as follows:
```
PlaNaVid/
├── model_zoo/
│   ├── uninavid-7b-full-224-video-fps-1-grid-2/  # Uni-NaVid checkpoint
│   └── eva_vit_g.pth                    # Visual encoder checkpoint
└── ...
```

Download the Uni-NaVid checkpoint from Hugging Face to `model_zoo`:

```
hf download Jzzhang/Uni-NaVid \
  --include "uninavid-7b-full-224-video-fps-1-grid-2/*" \
  --local-dir model_zoo
```

Download the EVA-ViT-G checkpoint to `model_zoo`:

```
wget -P model_zoo https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```
