# Source from Bash: source ~/projects/mini-sglang-kv/experiments/activate.sh
lab_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$lab_root/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$lab_root/.lab-tools:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$lab_root/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$lab_root"
unset lab_root
