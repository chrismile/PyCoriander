/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <map>

#include <c10/cuda/CUDAStream.h>
#include <nvrtc.h>

#include "CudaHelpers.hpp"
#include "MutualInformationCpu.hpp"
#include "MutualInformationKraskovHeader.hpp"

struct KernelCache {
    ~KernelCache() {
        if (cumodule) {
            checkCUresult(g_cudaDeviceApiFunctionTable.cuModuleUnload(cumodule), "Error in cuModuleUnload: ");
        }
    }
    std::map<std::string, std::string> preprocessorDefines;
    CUmodule cumodule{};
    CUfunction kernel{};
};

static KernelCache* kernelCache = nullptr;

void pycorianderCleanup() {
    if (kernelCache) {
        delete[] kernelCache;
        kernelCache = nullptr;
    }
    if (getIsCudaDeviceApiFunctionTableInitialized()) {
        freeCudaDeviceApiFunctionTable();
    }
}

torch::Tensor mutualInformationKraskovCuda(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k) {
    if (referenceTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in mutualInformationKraskovCuda: referenceTensor.sizes().size() > 2.");
    }
    if (queryTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in mutualInformationKraskovCuda: queryTensor.sizes().size() > 2.");
    }

    // Size of tensor: (M, N) or (N).
    const int64_t Mr = referenceTensor.sizes().size() == 1 ? 1 : referenceTensor.size(0);
    const int64_t Nr = referenceTensor.sizes().size() == 1 ? referenceTensor.size(0) : referenceTensor.size(1);
    const int64_t Mq = queryTensor.sizes().size() == 1 ? 1 : queryTensor.size(0);
    const int64_t Nq = queryTensor.sizes().size() == 1 ? queryTensor.size(0) : queryTensor.size(1);
    if (Nr != Nq || (Mr != Mq && Mr != 1 && Mq != 1)) {
        throw std::runtime_error("Error in mutualInformationKraskovCpu: Tensor size mismatch.");
    }
    const int64_t M = std::max(Mr, Mq);
    const int64_t N = Nr;

    if (!getIsCudaDeviceApiFunctionTableInitialized()) {
        initializeCudaDeviceApiFunctionTable();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor outputTensor = torch::zeros(
            M, at::TensorOptions().dtype(torch::kFloat32).device(referenceTensor.device()));

    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(N)));
    auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(N + 1)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));

    if (!kernelCache || kernelCache->preprocessorDefines != preprocessorDefines) {
        std::string code;
        for (const auto& entry : preprocessorDefines) {
            code += std::string("#define ") + entry.first + " " + entry.second + "\n";
        }
        code += "#line 1\n";
        code += std::string(reinterpret_cast<const char*>(MutualInformationKraskov_cu), MutualInformationKraskov_cu_len);

        nvrtcProgram prog;
        checkNvrtcResult(nvrtcCreateProgram(
                &prog, code.c_str(), "MutualInformationKraskov.cu", 0, nullptr, nullptr), "Error in nvrtcCreateProgram: ");
        auto retVal = nvrtcCompileProgram(prog, 0, nullptr);
        if (retVal == NVRTC_ERROR_COMPILATION) {
            size_t logSize = 0;
            checkNvrtcResult(nvrtcGetProgramLogSize(prog, &logSize), "Error in nvrtcGetProgramLogSize: ");
            char* log = new char[logSize];
            checkNvrtcResult(nvrtcGetProgramLog (prog, log), "Error in nvrtcGetProgramLog: ");
            std::cerr << "NVRTC log:" << std::endl << log << std::endl;
            delete[] log;
            checkNvrtcResult(nvrtcDestroyProgram(&prog), "Error in nvrtcDestroyProgram: ");
            exit(1);
        }

        size_t ptxSize = 0;
        checkNvrtcResult(nvrtcGetPTXSize(prog, &ptxSize), "Error in nvrtcGetPTXSize: ");
        char* ptx = new char[ptxSize];
        checkNvrtcResult(nvrtcGetPTX(prog, ptx), "Error in nvrtcGetPTX: ");
        checkNvrtcResult(nvrtcDestroyProgram(&prog), "Error in nvrtcDestroyProgram: ");

        if (kernelCache) {
            delete[] kernelCache;
        }
        kernelCache = new KernelCache;
        kernelCache->preprocessorDefines = preprocessorDefines;

        checkCUresult(g_cudaDeviceApiFunctionTable.cuModuleLoadDataEx(
                &kernelCache->cumodule, ptx, 0, nullptr, nullptr), "Error in cuModuleLoadDataEx: ");
        checkCUresult(g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &kernelCache->kernel, kernelCache->cumodule, "mutualInformationKraskov"), "Error in cuModuleGetFunction: ");
        delete[] ptx;
    }

    int minGridSize = 0;
    int bestBlockSize = 32;
    checkCUresult(g_cudaDeviceApiFunctionTable.cuOccupancyMaxPotentialBlockSize(
            &minGridSize, &bestBlockSize, kernelCache->kernel, nullptr, 0, 0), "Error in cuOccupancyMaxPotentialBlockSize: ");
    //std::cout << "minGridSize: " << minGridSize << ", bestBlockSize: " << bestBlockSize << std::endl;

    CUdevice cuDevice{};
    checkCUresult(g_cudaDeviceApiFunctionTable.cuDeviceGet(
            &cuDevice, at::cuda::current_device()), "Error in cuDeviceGet: ");
    CudaDeviceCoresInfo deviceCoresInfo = getNumCudaCores(cuDevice);
    //std::cout << "numMultiprocessors: " << deviceCoresInfo.numMultiprocessors << ", bestBlockSize: " << deviceCoresInfo.warpSize << std::endl;
    int BLOCK_SIZE = lastPowerOfTwo(bestBlockSize);
    while (BLOCK_SIZE > int(deviceCoresInfo.warpSize)
            && iceil(int(M), BLOCK_SIZE) * 2 < int(deviceCoresInfo.numMultiprocessors)) {
        BLOCK_SIZE /= 2;
    }
    //std::cout << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

    /*
     * Estimated time: M * N * log2(N).
     * On a RTX 3090, M = 1.76 * 10^6 and N = 100 takes approx. 1s.
     */
    const uint32_t numCudaCoresRtx3090 = 10496;
    double factorM = double(M) / (1.76 * 1e6);
    double factorN = double(N) / 100.0 * std::log2(double(N) / 100.0 + 1.0);
    double factorCudaCores = double(numCudaCoresRtx3090) / double(deviceCoresInfo.numCudaCoresTotal);
    auto batchCount = uint32_t(std::ceil(factorM * factorN * factorCudaCores));

    auto referenceArray = reinterpret_cast<CUdeviceptr>(referenceTensor.data_ptr());
    auto queryArray = reinterpret_cast<CUdeviceptr>(queryTensor.data_ptr());
    auto miArray = reinterpret_cast<CUdeviceptr>(outputTensor.data_ptr());
    auto referenceStride = Mr == 1 ? 0 : uint32_t(referenceTensor.stride(0));
    auto queryStride = Mq == 1 ? 0 : uint32_t(queryTensor.stride(0));
    auto batchSize = uint32_t(M);
    if (batchCount == 1) {
        auto batchOffset = uint32_t(0);
        void* kernelParameters[] = {
                &referenceArray, &queryArray, &miArray, &referenceStride, &queryStride, &batchOffset, &batchSize
        };
        checkCUresult(g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                kernelCache->kernel, iceil(int(M), BLOCK_SIZE), 1, 1, //< Grid size.
                BLOCK_SIZE, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr
        ), "Error in cuLaunchKernel: "); //< Extra (empty).
    } else {
        auto batchSizeLocal = uiceil(uint32_t(M), batchCount);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            auto batchOffset = batchSizeLocal * batchIdx;
            if (batchOffset + batchSizeLocal > uint32_t(M)) {
                batchSizeLocal = uint32_t(M) - batchSizeLocal;
            }
            void* kernelParameters[] = {
                    &referenceArray, &queryArray, &miArray, &referenceStride, &queryStride, &batchOffset, &batchSize
            };
            checkCUresult(g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                    kernelCache->kernel, iceil(int(batchSizeLocal), BLOCK_SIZE), 1, 1, //< Grid size.
                    BLOCK_SIZE, 1, 1, //< Block size.
                    0, //< Dynamic shared memory size.
                    stream,
                    kernelParameters, //< Kernel parameters.
                    nullptr
            ), "Error in cuLaunchKernel: "); //< Extra (empty).
        }
    }

    return outputTensor;
}
