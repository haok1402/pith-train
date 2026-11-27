#include <fstream>
#include <iostream>
#include <iterator>

#include <csrc/runtime/jit.hpp>
#include <csrc/jit_kernels/heuristics/sm90.hpp>

extern "C" cudaError_t cudaGetDevice(int* index) {
    *index = 0;
    return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* pointer) {
    return pointer == nullptr ? cudaSuccess : cudaErrorInvalidValue;
}
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int index) {
    if (index != 0)
        return cudaErrorInvalidDevice;
    *prop = {};
    prop->major = 9;
    prop->minor = 0;
    prop->multiProcessorCount = 132;
    return cudaSuccess;
}

int main(int argc, char** argv) {
    using namespace deep_gemm;
    if (argc != 4)
        return 2;
    init_jit(argv[1]);
    std::ifstream source(argv[2]);
    std::ifstream shapes(argv[3]);
    if (!source || !shapes)
        return 2;
    const std::string format{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
    int n, k;
    while (shapes >> n >> k) {
        constexpr int m = 32768;
        const GemmDesc desc{
            .gemm_type = GemmType::Normal, .kernel_type = KernelType::Kernel1D2D,
            .m = m, .n = n, .k = k, .num_groups = 1,
            .a_dtype = torch::kFloat8_e4m3fn, .b_dtype = torch::kFloat8_e4m3fn, .cd_dtype = torch::kBFloat16,
            .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
            .with_accumulation = false, .num_sms = 132, .tc_util = 100, .compiled_dims = "nk",
        };
        const auto config = get_best_config<SM90ArchSpec>(desc);
        const auto render = [&](auto&&... args) {
            return std::vformat(format, std::make_format_args(args...));
        };
        const auto code = render(
            "cute::UMMA::Major::K", 0, n, k,
            desc.num_groups,
            config.layout.block_m, config.layout.block_n, config.layout.block_k,
            config.storage_config.swizzle_a_mode, config.storage_config.swizzle_b_mode, config.storage_config.swizzle_cd_mode,
            config.pipeline_config.num_stages,
            config.launch_config.num_tma_threads, config.launch_config.num_math_threads,
            config.layout.get_cluster_size(), config.layout.cluster_n > 1,
            config.launch_config.num_sms, "GemmType::Normal", "cutlass::bfloat16_t");
        const auto artifact = jit->compile_without_load("sm90_fp8_gemm_1d2d", code);
        std::cout << "DG_TRAIN " << n << " " << k << " " << artifact.string() << std::endl;
    }
}
