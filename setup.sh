#!/bin/bash
conda activate annyfit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/submodules/multi-hmr:${REPO_ROOT}/submodules/CameraHMR"
export PYOPENGL_PLATFORM=egl
