#!/usr/bin/env bash
# Usage: bash install.sh [--env-name annyfit]
set -e

ENV_NAME="annyfit"
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# numpy<2 early — some deps (chumpy, mmcv) are incompatible with numpy 2.x
pip install "numpy<2"

# mmcv (needs --no-build-isolation for setuptools/pkg_resources)
pip install mmcv==1.3.9 --no-build-isolation \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.5/index.html

# chumpy (ViTPose dependency, needs git install + --no-build-isolation)
pip install --no-build-isolation git+https://github.com/mattloper/chumpy

# ViTPose
# Patch model_split.py: fix bug where 'expert' is checked against all keys instead of current key
sed -i "s/if 'expert' in keys:/if 'expert' in key:/" submodules/ViTPose/tools/model_split.py
pip install -v -e submodules/ViTPose
pip install --no-build-isolation albumentations  # before requirements.txt: needs pkg_resources
pip install -r submodules/ViTPose/requirements.txt

# anny
pip install "anny[warp,examples]@git+https://github.com/naver/anny.git"

# detectron2 (CameraHMR dependency)
pip install --no-build-isolation 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git'

# Other dependencies
pip install -r requirements.txt

# Submodules
pip install -e submodules/Grounded-SAM-2
pip install --no-build-isolation -e submodules/Grounded-SAM-2/grounding_dino
pip install "git+https://github.com/laubravo/UniDepth.git"
# Patch UniDepth: PEP 604 union syntax (int | tuple) breaks torch JIT
GEOM_PY="$(python -c 'import unidepth.utils.geometric as m; print(m.__file__)')"
sed -i 's/from typing import Tuple/from typing import Tuple, Union/' "$GEOM_PY"
sed -i 's/kernel_size: int | tuple\[int, int\]/kernel_size: Union[int, Tuple[int, int]]/g' "$GEOM_PY"

echo "Done. Run: bash scripts/download_checkpoints.sh && source setup.sh"
