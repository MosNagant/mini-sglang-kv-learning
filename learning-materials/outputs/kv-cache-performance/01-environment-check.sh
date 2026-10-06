#!/usr/bin/env bash
# Read-only environment checks, plus a timestamped log in the user's Linux home.
# This script does not install packages, change the environment, or download models.
{
    date -Is
    uname -r
    cat /etc/os-release

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        /usr/lib/wsl/lib/nvidia-smi
    fi

    for tool in nvcc python3 uv git ncu nsys; do
        printf '\n--- %s ---\n' "$tool"
        if command -v "$tool"; then
            "$tool" --version
        else
            printf '%s: NOT_ON_PATH\n' "$tool"
        fi
    done

    df -h "$HOME" /mnt/c

    if command -v git >/dev/null 2>&1; then
        timeout 20s git ls-remote https://github.com/sgl-project/mini-sglang.git refs/heads/main ||
            printf 'REMOTE_CHECK_FAILED: keep the error above; commit is not yet verified.\n'
    fi
} 2>&1 | tee "$HOME/cuda-env-check-$(date +%Y%m%d-%H%M%S).txt"
