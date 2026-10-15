"""Install the pinned evaluation and training stack on a Colab GPU runtime.

Run on the Colab machine (infra/colab/run_on_colab.py does this for you). PyTorch comes from
the Colab image (2.11.0+cu130 when these experiments ran). causal-conv1d is built from source
for the T4 only (TORCH_CUDA_ARCH_LIST=7.5); a build for every architecture takes over 45 minutes.

Important: start the causal-conv1d build when no large model is loaded. Building it while a
9B model trains exhausted the 12.7 GB of host RAM of a Colab T4 machine in our runs.
"""
import os
import shutil
import subprocess
import sys

PACKAGES = [
    "laya==0.4.2",
    "transformers==5.19.0",
    "bitsandbytes==0.50.2",
    "accelerate==1.15.0",
    "peft==0.21.0",
    "sentencepiece",
    "jsonschema==4.26.0",
    "flash-linear-attention==0.5.2",
    "lm-format-enforcer @ git+https://github.com/noamgat/lm-format-enforcer.git@817f944fcc8851917c40300a47f62d1defc3ffc3",
]


def main():
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--system", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]) + PACKAGES
    subprocess.check_call(cmd)
    if "--with-causal-conv1d" in sys.argv:
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5"),
                   MAX_JOBS=os.environ.get("MAX_JOBS", "2"), CAUSAL_CONV1D_FORCE_BUILD="TRUE")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "causal-conv1d==1.7.0"], env=env)
    import torch
    print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    main()
