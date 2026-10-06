#!/usr/bin/env bash
# Run as the Linux user weltlaw, not with sudo bash. Only package operations use sudo.
set -Eeuo pipefail
export LC_ALL=C.UTF-8

cuda_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cuda_stamp=$(date +%Y%m%d-%H%M%S)
cuda_log_dir="$cuda_source_dir/logs"
mkdir -p -- "$cuda_log_dir"
cuda_log="$cuda_log_dir/cuda-setup-$cuda_stamp.log"
cuda_status="$cuda_log_dir/cuda-latest-status.txt"
exec > >(tee -a "$cuda_log") 2>&1

fail() { printf '\nERROR: %s\n' "$*" >&2; return 1; }
on_error() {
    local code=$1 line=$2
    trap - ERR
    printf '\nSETUP STOPPED at line %s (exit %s). Log: %s\n' "$line" "$code" "$cuda_log"
    printf 'INCOMPLETE: CUDA setup stopped at line %s; see %s\n' "$line" "$cuda_log" > "$cuda_status"
    exit "$code"
}
trap 'on_error "$?" "$LINENO"' ERR
printf 'IN PROGRESS: %s\n' "$cuda_log" > "$cuda_status"

printf '\n[1/6] Checking Ubuntu, Linux user, GPU, and free space\n'
[[ $EUID -ne 0 ]] || fail 'Run as weltlaw. The script asks for sudo only when needed.'
[[ $(id -un) == weltlaw ]] || fail 'This setup is for the Linux account weltlaw.'
[[ ${WSL_DISTRO_NAME:-} == CUDA-Ubuntu-24.04 ]] || fail 'Run inside CUDA-Ubuntu-24.04.'
grep -qi 'microsoft.*WSL2' /proc/sys/kernel/osrelease || fail 'A running WSL 2 kernel is required.'
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || fail 'This setup expects Ubuntu 24.04.'
[[ $(uname -m) == x86_64 ]] || fail 'This setup expects x86_64.'
printf 'User: %s\nHome: %s\nOS: %s\nKernel: %s\n' "$(id -un)" "$HOME" "$PRETTY_NAME" "$(uname -r)"
[[ -x /usr/lib/wsl/lib/nvidia-smi ]] || fail 'The Windows GPU driver is not available inside WSL.'
/usr/lib/wsl/lib/nvidia-smi
cuda_driver=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')
dpkg --compare-versions "$cuda_driver" ge 572.61 || fail "Driver $cuda_driver is below the selected Windows baseline 572.61."
df -h / "$HOME" /mnt/c
cuda_free_kb=$(df -Pk / | awk 'END {print $4}')
cuda_host_free_kb=$(df -Pk /mnt/c | awk 'END {print $4}')
[[ $cuda_free_kb -ge 20971520 && $cuda_host_free_kb -ge 20971520 ]] || fail 'Keep at least 20 GiB free in Ubuntu and on Windows C: before installing the full toolkit.'
printf 'Existing compiler: '
command -v nvcc || true
[[ -d "$cuda_source_dir/cuda-starter" ]] || fail 'The cuda-starter template is missing.'

printf '\n[2/6] Installing build tools from Ubuntu repositories\n'
printf 'Enter the password you created for weltlaw if sudo asks. Password input is invisible.\n'
sudo -v
cuda_apt_options=(-o Acquire::ForceIPv4=true -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o DPkg::Lock::Timeout=120)
sudo apt-get "${cuda_apt_options[@]}" -o APT::Update::Error-Mode=any update
sudo apt-get "${cuda_apt_options[@]}" --no-remove install -y --no-install-recommends build-essential cmake ninja-build git curl ca-certificates

printf '\n[3/6] Configuring the official NVIDIA WSL repository\n'
cuda_cache="$HOME/.cache/cuda-wsl-setup"
mkdir -p -- "$cuda_cache"
cuda_keyring="$cuda_cache/cuda-keyring_1.1-1_all.deb"
cuda_keyring_url='https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb'
if ! curl --ipv4 --fail --location --show-error --retry 2 --connect-timeout 15 --max-time 120 --output "$cuda_keyring" "$cuda_keyring_url"; then
    fail 'Cannot download the NVIDIA repository keyring from inside WSL. Keep this log; the localhost proxy/network may need configuration.'
fi
[[ $(dpkg-deb -f "$cuda_keyring" Package) == cuda-keyring ]] || fail 'Unexpected repository package name.'
[[ $(dpkg-deb -f "$cuda_keyring" Version) == 1.1-1 ]] || fail 'Unexpected repository package version.'
cuda_installed_keyring=$(dpkg-query -W -f='${Version}' cuda-keyring 2>/dev/null || true)
if [[ -n $cuda_installed_keyring ]] && dpkg --compare-versions "$cuda_installed_keyring" gt 1.1-1; then
    fail 'A newer CUDA keyring is already installed; review its repository before changing it.'
fi
sudo dpkg -i "$cuda_keyring"
sudo apt-get "${cuda_apt_options[@]}" -o APT::Update::Error-Mode=any update

printf '\n[4/6] Installing CUDA Toolkit 12.8 (large download; keep this window open)\n'
cuda_candidate=$(apt-cache policy cuda-toolkit-12-8 | awk '/Candidate:/ {print $2}')
[[ $cuda_candidate == 12.8.* ]] || fail 'The NVIDIA repository did not provide a CUDA 12.8 toolkit candidate.'
cuda_plan="$cuda_log_dir/cuda-package-plan-$cuda_stamp.txt"
sudo apt-get "${cuda_apt_options[@]}" --simulate --no-remove --no-install-recommends install "cuda-toolkit-12-8=$cuda_candidate" | tee "$cuda_plan"
if grep -Eq '^Inst (cuda-drivers|nvidia-driver|nvidia-dkms|nvidia-kernel|nvidia-headless|libnvidia-compute|linux-modules-nvidia)' "$cuda_plan"; then
    fail 'The package plan unexpectedly includes Linux GPU driver packages; installation stopped.'
fi
sudo apt-get "${cuda_apt_options[@]}" --no-remove install -y --no-install-recommends "cuda-toolkit-12-8=$cuda_candidate"
[[ -x /usr/local/cuda-12.8/bin/nvcc ]] || fail 'nvcc was not installed at the expected path.'

printf '\n[5/6] Configuring the shell and creating a starter project\n'
cuda_path_line='export PATH="/usr/local/cuda-12.8/bin:$PATH"'
touch "$HOME/.bashrc"
if ! grep -Fqx "$cuda_path_line" "$HOME/.bashrc"; then
    cp -p -- "$HOME/.bashrc" "$HOME/.bashrc.before-cuda-$cuda_stamp"
    printf '\n# CUDA Toolkit 12.8\n%s\n' "$cuda_path_line" >> "$HOME/.bashrc"
fi
export PATH="/usr/local/cuda-12.8/bin:$PATH"
nvcc --version
cmake --version
gcc --version
cuda_project="$HOME/projects/cuda-starter"
if [[ -e $cuda_project || -L $cuda_project ]]; then
    cuda_project="$HOME/projects/cuda-starter-$cuda_stamp-$$"
    printf 'An existing cuda-starter is preserved. Using %s\n' "$cuda_project"
fi
mkdir -p -- "$cuda_project"
cp -R -- "$cuda_source_dir/cuda-starter/." "$cuda_project/"

printf '\n[6/6] Compiling and running the GPU calculation\n'
cmake -S "$cuda_project" -B "$cuda_project/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build "$cuda_project/build" --parallel 4
ctest --test-dir "$cuda_project/build" --output-on-failure --verbose
printf 'PASS: CUDA Toolkit 12.8 compiled a CUDA kernel; GPU vector addition matched the CPU reference.\nProject: %s\nLog: %s\n' "$cuda_project" "$cuda_log" > "$cuda_status"
cat "$cuda_status"
printf '\nReopen Ubuntu so the nvcc PATH takes effect in your interactive terminal.\n'
printf 'Then: cd %q\n' "$cuda_project"
printf 'Run again: ./build/vector_add\n'
