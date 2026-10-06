# 第一个 CUDA 项目

这个程序把两个数组从 CPU 内存复制到 GPU，在 GPU 上并行相加，再把结果复制回来，逐项与 CPU 计算结果比较。

`vector_add.cu` 中带 `__global__` 的函数是 GPU kernel；`<<<blocks, threads>>>` 启动 GPU 线程；`cudaMemcpy` 负责 CPU 与 GPU 之间的数据传输。

## 编译运行

在 Ubuntu 中进入本项目目录后执行：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --parallel 4
./build/vector_add
```

成功时会显示 GPU 名称和 `PASS: all 1048589 GPU results match the CPU reference.`。

也可以通过 CTest 验证：

```bash
ctest --test-dir build --output-on-failure --verbose
```

CMake 中 `CUDA_ARCHITECTURES 89` 对应本机 RTX 4070 Ti。更换显卡后需按目标 GPU 调整。

安装脚本只在新目录中放置示例；已有同名项目会保留。你可以修改 kernel 继续练习，修改后重新编译即可。
