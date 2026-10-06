#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t result, const char* call, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error on line %d: %s: %s\n",
                     line, call, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

__global__ void vector_add(const float* a, const float* b, float* c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "No CUDA-capable GPU was found.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp gpu{};
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s | Compute capability: %d.%d\n", gpu.name, gpu.major, gpu.minor);

    // The extra 13 elements also exercise the bounds check in the last block.
    constexpr int count = (1 << 20) + 13;
    constexpr size_t bytes = static_cast<size_t>(count) * sizeof(float);
    std::vector<float> a(count), b(count), result(count);
    for (int i = 0; i < count; ++i) {
        a[i] = static_cast<float>((i % 1001) - 500) * 0.25f;
        b[i] = static_cast<float>((i % 257) - 128) * 0.5f;
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes));
    CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
    // NaNs ensure an unwritten output cannot accidentally pass verification.
    CUDA_CHECK(cudaMemset(device_c, 0xff, bytes));

    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(device_a, device_b, device_c, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    for (int i = 0; i < count; ++i) {
        const float expected = a[i] + b[i];
        if (!std::isfinite(result[i]) || std::fabs(result[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "FAIL at index %d: GPU=%g, CPU=%g\n", i, result[i], expected);
            return EXIT_FAILURE;
        }
    }
    std::printf("PASS: all %d GPU results match the CPU reference.\n", count);
    return EXIT_SUCCESS;
}
