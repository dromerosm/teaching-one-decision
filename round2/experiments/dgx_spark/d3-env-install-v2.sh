#!/bin/bash
# v2: same packages on a uv-managed standalone CPython 3.12, which ships the C headers Triton needs
# (the system python3.12 has no Python.h, so Triton could not build its CUDA driver helper in v1).
set -euo pipefail
ROOT=$HOME/d3-eval-20261012
UV=$HOME/.local/bin/uv
LOG=$HOME/dgx-ops/d3-env-v2-$(date +%F_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "D3_ENV2_START $(date -Is) log=$LOG"
test ! -e "$ROOT/venv-managed" || { echo "ABORT: venv-managed exists"; exit 3; }
"$UV" python install 3.12
"$UV" venv --managed-python --python 3.12 "$ROOT/venv-managed"
PY="$ROOT/venv-managed/bin/python"
"$PY" -c "import sysconfig,os;h=os.path.join(sysconfig.get_paths()['include'],'Python.h');print('PYTHON_H',h,os.path.exists(h))"
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchvision==0.26.0
"$UV" pip install --python "$PY" transformers==5.17.0 accelerate==1.15.0 "safetensors>=0.6" "huggingface_hub>=1.0" \
  flash-linear-attention==0.5.2 einops "pillow>=10" opencv-python-headless
"$PY" - <<'PY'
import torch, triton, triton.language as tl, transformers, fla
@triton.jit
def add(x, y, out, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
add[(4,)](x, y, o, 4096, B=1024)
assert torch.allclose(o, x + y)
print("TRITON_KERNEL_PASS", triton.__version__, torch.__version__, torch.cuda.get_device_capability(0), transformers.__version__)
PY
"$UV" pip freeze --python "$PY" > "$ROOT/pip-freeze-managed.txt"
echo "D3_ENV2_EXIT=0"
