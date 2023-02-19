/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Christoph Neuhauser
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

#include <exception>

#include "CudaHelpers.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

CudaDeviceApiFunctionTable g_cudaDeviceApiFunctionTable{};

#ifdef _WIN32
HMODULE g_cudaLibraryHandle = nullptr;
#define dlsym GetProcAddress
#else
void* g_cudaLibraryHandle = nullptr;
#endif

bool initializeCudaDeviceApiFunctionTable() {
    typedef CUresult ( *PFN_cuInit )( unsigned int Flags );
    typedef CUresult ( *PFN_cuGetErrorString )( CUresult error, const char **pStr );
    typedef CUresult ( *PFN_cuDeviceGet )(CUdevice *device, int ordinal);
    typedef CUresult ( *PFN_cuDeviceGetCount )(int *count);
    typedef CUresult ( *PFN_cuDeviceGetUuid )(CUuuid *uuid, CUdevice dev);
    typedef CUresult ( *PFN_cuDeviceGetAttribute )(int *pi, CUdevice_attribute attrib, CUdevice dev);
    typedef CUresult ( *PFN_cuCtxCreate )( CUcontext *pctx, unsigned int flags, CUdevice dev );
    typedef CUresult ( *PFN_cuCtxDestroy )( CUcontext ctx );
    typedef CUresult ( *PFN_cuStreamCreate )( CUstream *phStream, unsigned int Flags );
    typedef CUresult ( *PFN_cuStreamDestroy )( CUstream hStream );
    typedef CUresult ( *PFN_cuStreamSynchronize )( CUstream hStream );
    typedef CUresult ( *PFN_cuMemAlloc )( CUdeviceptr *dptr, size_t bytesize );
    typedef CUresult ( *PFN_cuMemFree )( CUdeviceptr dptr );
    typedef CUresult ( *PFN_cuMemcpyDtoH )( void *dstHost, CUdeviceptr srcDevice, size_t ByteCount );
    typedef CUresult ( *PFN_cuMemcpyHtoD )( CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount );
    typedef CUresult ( *PFN_cuMemAllocAsync )( CUdeviceptr *dptr, size_t bytesize, CUstream hStream );
    typedef CUresult ( *PFN_cuMemFreeAsync )( CUdeviceptr dptr, CUstream hStream );
    typedef CUresult ( *PFN_cuMemsetD8Async )( CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream );
    typedef CUresult ( *PFN_cuMemsetD16Async )( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream );
    typedef CUresult ( *PFN_cuMemsetD32Async )( CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream );
    typedef CUresult ( *PFN_cuMemcpyAsync )( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream );
    typedef CUresult ( *PFN_cuMemcpyDtoHAsync )( void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
    typedef CUresult ( *PFN_cuMemcpyHtoDAsync )( CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream );
    typedef CUresult ( *PFN_cuMemcpy2DAsync )( const CUDA_MEMCPY2D* pCopy, CUstream hStream );
    typedef CUresult ( *PFN_cuMemcpy3DAsync )( const CUDA_MEMCPY3D* pCopy, CUstream hStream );
    typedef CUresult ( *PFN_cuArrayCreate )( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    typedef CUresult ( *PFN_cuArray3DCreate )( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
    typedef CUresult ( *PFN_cuArrayDestroy )( CUarray hArray );
    typedef CUresult ( *PFN_cuMipmappedArrayCreate )( CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels );
    typedef CUresult ( *PFN_cuMipmappedArrayDestroy )( CUmipmappedArray hMipmappedArray );
    typedef CUresult ( *PFN_cuMipmappedArrayGetLevel )( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level );
    typedef CUresult ( *PFN_cuTexObjectCreate )( CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc );
    typedef CUresult ( *PFN_cuTexObjectDestroy )( CUtexObject texObject );
    typedef CUresult ( *PFN_cuSurfObjectCreate )( CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc );
    typedef CUresult ( *PFN_cuSurfObjectDestroy )( CUsurfObject surfObject );
    typedef CUresult ( *PFN_cuImportExternalMemory )( CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc );
    typedef CUresult ( *PFN_cuExternalMemoryGetMappedBuffer )( CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc );
    typedef CUresult ( *PFN_cuExternalMemoryGetMappedMipmappedArray )( CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc );
    typedef CUresult ( *PFN_cuDestroyExternalMemory )( CUexternalMemory extMem );
    typedef CUresult ( *PFN_cuImportExternalSemaphore )( CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc );
    typedef CUresult ( *PFN_cuSignalExternalSemaphoresAsync )( const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream );
    typedef CUresult ( *PFN_cuWaitExternalSemaphoresAsync )( const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream );
    typedef CUresult ( *PFN_cuDestroyExternalSemaphore )( CUexternalSemaphore extSem );
    typedef CUresult ( *PFN_cuModuleLoad )( CUmodule* module, const char* fname );
    typedef CUresult ( *PFN_cuModuleLoadData )( CUmodule* module, const void* image );
    typedef CUresult ( *PFN_cuModuleLoadDataEx )( CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues );
    typedef CUresult ( *PFN_cuModuleLoadFatBinary )( CUmodule* module, const void* fatCubin );
    typedef CUresult ( *PFN_cuModuleUnload )( CUmodule hmod );
    typedef CUresult ( *PFN_cuModuleGetFunction )( CUfunction* hfunc, CUmodule hmod, const char* name );
    typedef CUresult ( *PFN_cuModuleGetGlobal )( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name );
    typedef CUresult ( *PFN_cuLaunchKernel )( CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra );
    typedef CUresult ( *PFN_cuOccupancyMaxPotentialBlockSize )( int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit );

#if defined(__linux__)
    g_cudaLibraryHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cudaLibraryHandle) {
        throw std::runtime_error("initializeCudaDeviceApiFunctionTable: Could not load libcuda.so.");
        return false;
    }
#elif defined(_WIN32)
    g_cudaLibraryHandle = LoadLibraryA("nvcuda.dll");
    if (!g_cudaLibraryHandle) {
        throw std::runtime_error("initializeCudaDeviceApiFunctionTable: Could not load nvcuda.dll.");
        return false;
    }
#endif
    g_cudaDeviceApiFunctionTable.cuInit = PFN_cuInit(dlsym(g_cudaLibraryHandle, TOSTRING(cuInit)));
    g_cudaDeviceApiFunctionTable.cuGetErrorString = PFN_cuGetErrorString(dlsym(g_cudaLibraryHandle, TOSTRING(cuGetErrorString)));
    g_cudaDeviceApiFunctionTable.cuDeviceGet = PFN_cuDeviceGet(dlsym(g_cudaLibraryHandle, TOSTRING(cuDeviceGet)));
    g_cudaDeviceApiFunctionTable.cuDeviceGetCount = PFN_cuDeviceGetCount(dlsym(g_cudaLibraryHandle, TOSTRING(cuDeviceGetCount)));
    g_cudaDeviceApiFunctionTable.cuDeviceGetUuid = PFN_cuDeviceGetUuid(dlsym(g_cudaLibraryHandle, TOSTRING(cuDeviceGetUuid)));
    g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute = PFN_cuDeviceGetAttribute(dlsym(g_cudaLibraryHandle, TOSTRING(cuDeviceGetAttribute)));
    g_cudaDeviceApiFunctionTable.cuCtxCreate = PFN_cuCtxCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuCtxCreate)));
    g_cudaDeviceApiFunctionTable.cuCtxDestroy = PFN_cuCtxDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuCtxDestroy)));
    g_cudaDeviceApiFunctionTable.cuStreamCreate = PFN_cuStreamCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuStreamCreate)));
    g_cudaDeviceApiFunctionTable.cuStreamDestroy = PFN_cuStreamDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuStreamDestroy)));
    g_cudaDeviceApiFunctionTable.cuStreamSynchronize = PFN_cuStreamSynchronize(dlsym(g_cudaLibraryHandle, TOSTRING(cuStreamSynchronize)));
    g_cudaDeviceApiFunctionTable.cuMemAlloc = PFN_cuMemAlloc(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemAlloc)));
    g_cudaDeviceApiFunctionTable.cuMemFree = PFN_cuMemFree(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemFree)));
    g_cudaDeviceApiFunctionTable.cuMemcpyDtoH = PFN_cuMemcpyDtoH(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpyDtoH)));
    g_cudaDeviceApiFunctionTable.cuMemcpyHtoD = PFN_cuMemcpyHtoD(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpyHtoD)));
    g_cudaDeviceApiFunctionTable.cuMemAllocAsync = PFN_cuMemAllocAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemAllocAsync)));
    g_cudaDeviceApiFunctionTable.cuMemFreeAsync = PFN_cuMemFreeAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemFreeAsync)));
    g_cudaDeviceApiFunctionTable.cuMemsetD8Async = PFN_cuMemsetD8Async(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemsetD8Async)));
    g_cudaDeviceApiFunctionTable.cuMemsetD16Async = PFN_cuMemsetD16Async(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemsetD16Async)));
    g_cudaDeviceApiFunctionTable.cuMemsetD32Async = PFN_cuMemsetD32Async(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemsetD32Async)));
    g_cudaDeviceApiFunctionTable.cuMemcpyAsync = PFN_cuMemcpyAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpyAsync)));
    g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync = PFN_cuMemcpyDtoHAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpyDtoHAsync)));
    g_cudaDeviceApiFunctionTable.cuMemcpyHtoDAsync = PFN_cuMemcpyHtoDAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpyHtoDAsync)));
    g_cudaDeviceApiFunctionTable.cuMemcpy2DAsync = PFN_cuMemcpy2DAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpy2DAsync)));
    g_cudaDeviceApiFunctionTable.cuMemcpy3DAsync = PFN_cuMemcpy3DAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuMemcpy3DAsync)));
    g_cudaDeviceApiFunctionTable.cuArrayCreate = PFN_cuArrayCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuArrayCreate)));
    g_cudaDeviceApiFunctionTable.cuArray3DCreate = PFN_cuArray3DCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuArray3DCreate)));
    g_cudaDeviceApiFunctionTable.cuArrayDestroy = PFN_cuArrayDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuArrayDestroy)));
    g_cudaDeviceApiFunctionTable.cuMipmappedArrayCreate = PFN_cuMipmappedArrayCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuMipmappedArrayCreate)));
    g_cudaDeviceApiFunctionTable.cuMipmappedArrayDestroy = PFN_cuMipmappedArrayDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuMipmappedArrayDestroy)));
    g_cudaDeviceApiFunctionTable.cuMipmappedArrayGetLevel = PFN_cuMipmappedArrayGetLevel(dlsym(g_cudaLibraryHandle, TOSTRING(cuMipmappedArrayGetLevel)));
    g_cudaDeviceApiFunctionTable.cuTexObjectCreate = PFN_cuTexObjectCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuTexObjectCreate)));
    g_cudaDeviceApiFunctionTable.cuTexObjectDestroy = PFN_cuTexObjectDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuTexObjectDestroy)));
    g_cudaDeviceApiFunctionTable.cuSurfObjectCreate = PFN_cuSurfObjectCreate(dlsym(g_cudaLibraryHandle, TOSTRING(cuSurfObjectCreate)));
    g_cudaDeviceApiFunctionTable.cuSurfObjectDestroy = PFN_cuSurfObjectDestroy(dlsym(g_cudaLibraryHandle, TOSTRING(cuSurfObjectDestroy)));
    g_cudaDeviceApiFunctionTable.cuImportExternalMemory = PFN_cuImportExternalMemory(dlsym(g_cudaLibraryHandle, TOSTRING(cuImportExternalMemory)));
    g_cudaDeviceApiFunctionTable.cuExternalMemoryGetMappedBuffer = PFN_cuExternalMemoryGetMappedBuffer(dlsym(g_cudaLibraryHandle, TOSTRING(cuExternalMemoryGetMappedBuffer)));
    g_cudaDeviceApiFunctionTable.cuExternalMemoryGetMappedMipmappedArray = PFN_cuExternalMemoryGetMappedMipmappedArray(dlsym(g_cudaLibraryHandle, TOSTRING(cuExternalMemoryGetMappedMipmappedArray)));
    g_cudaDeviceApiFunctionTable.cuDestroyExternalMemory = PFN_cuDestroyExternalMemory(dlsym(g_cudaLibraryHandle, TOSTRING(cuDestroyExternalMemory)));
    g_cudaDeviceApiFunctionTable.cuImportExternalSemaphore = PFN_cuImportExternalSemaphore(dlsym(g_cudaLibraryHandle, TOSTRING(cuImportExternalSemaphore)));
    g_cudaDeviceApiFunctionTable.cuSignalExternalSemaphoresAsync = PFN_cuSignalExternalSemaphoresAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuSignalExternalSemaphoresAsync)));
    g_cudaDeviceApiFunctionTable.cuWaitExternalSemaphoresAsync = PFN_cuWaitExternalSemaphoresAsync(dlsym(g_cudaLibraryHandle, TOSTRING(cuWaitExternalSemaphoresAsync)));
    g_cudaDeviceApiFunctionTable.cuDestroyExternalSemaphore = PFN_cuDestroyExternalSemaphore(dlsym(g_cudaLibraryHandle, TOSTRING(cuDestroyExternalSemaphore)));
    g_cudaDeviceApiFunctionTable.cuModuleLoad = PFN_cuModuleLoad(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleLoad)));
    g_cudaDeviceApiFunctionTable.cuModuleLoadData = PFN_cuModuleLoadData(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleLoadData)));
    g_cudaDeviceApiFunctionTable.cuModuleLoadDataEx = PFN_cuModuleLoadDataEx(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleLoadDataEx)));
    g_cudaDeviceApiFunctionTable.cuModuleLoadFatBinary = PFN_cuModuleLoadFatBinary(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleLoadFatBinary)));
    g_cudaDeviceApiFunctionTable.cuModuleUnload = PFN_cuModuleUnload(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleUnload)));
    g_cudaDeviceApiFunctionTable.cuModuleGetFunction = PFN_cuModuleGetFunction(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleGetFunction)));
    g_cudaDeviceApiFunctionTable.cuModuleGetGlobal = PFN_cuModuleGetGlobal(dlsym(g_cudaLibraryHandle, TOSTRING(cuModuleGetGlobal)));
    g_cudaDeviceApiFunctionTable.cuLaunchKernel = PFN_cuLaunchKernel(dlsym(g_cudaLibraryHandle, TOSTRING(cuLaunchKernel)));
    g_cudaDeviceApiFunctionTable.cuOccupancyMaxPotentialBlockSize = PFN_cuOccupancyMaxPotentialBlockSize(dlsym(g_cudaLibraryHandle, TOSTRING(cuOccupancyMaxPotentialBlockSize)));

    if (!g_cudaDeviceApiFunctionTable.cuInit
            || !g_cudaDeviceApiFunctionTable.cuGetErrorString
            || !g_cudaDeviceApiFunctionTable.cuDeviceGet
            || !g_cudaDeviceApiFunctionTable.cuDeviceGetCount
            || !g_cudaDeviceApiFunctionTable.cuDeviceGetUuid
            || !g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute
            || !g_cudaDeviceApiFunctionTable.cuCtxCreate
            || !g_cudaDeviceApiFunctionTable.cuCtxDestroy
            || !g_cudaDeviceApiFunctionTable.cuStreamCreate
            || !g_cudaDeviceApiFunctionTable.cuStreamDestroy
            || !g_cudaDeviceApiFunctionTable.cuStreamSynchronize
            || !g_cudaDeviceApiFunctionTable.cuMemAlloc
            || !g_cudaDeviceApiFunctionTable.cuMemFree
            || !g_cudaDeviceApiFunctionTable.cuMemcpyDtoH
            || !g_cudaDeviceApiFunctionTable.cuMemcpyHtoD
            || !g_cudaDeviceApiFunctionTable.cuMemAllocAsync
            || !g_cudaDeviceApiFunctionTable.cuMemFreeAsync
            || !g_cudaDeviceApiFunctionTable.cuMemsetD8Async
            || !g_cudaDeviceApiFunctionTable.cuMemsetD16Async
            || !g_cudaDeviceApiFunctionTable.cuMemsetD32Async
            || !g_cudaDeviceApiFunctionTable.cuMemcpyAsync
            || !g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync
            || !g_cudaDeviceApiFunctionTable.cuMemcpyHtoDAsync
            || !g_cudaDeviceApiFunctionTable.cuMemcpy2DAsync
            || !g_cudaDeviceApiFunctionTable.cuMemcpy3DAsync
            || !g_cudaDeviceApiFunctionTable.cuArrayCreate
            || !g_cudaDeviceApiFunctionTable.cuArray3DCreate
            || !g_cudaDeviceApiFunctionTable.cuArrayDestroy
            || !g_cudaDeviceApiFunctionTable.cuMipmappedArrayCreate
            || !g_cudaDeviceApiFunctionTable.cuMipmappedArrayDestroy
            || !g_cudaDeviceApiFunctionTable.cuMipmappedArrayGetLevel
            || !g_cudaDeviceApiFunctionTable.cuTexObjectCreate
            || !g_cudaDeviceApiFunctionTable.cuTexObjectDestroy
            || !g_cudaDeviceApiFunctionTable.cuSurfObjectCreate
            || !g_cudaDeviceApiFunctionTable.cuSurfObjectDestroy
            || !g_cudaDeviceApiFunctionTable.cuImportExternalMemory
            || !g_cudaDeviceApiFunctionTable.cuExternalMemoryGetMappedBuffer
            || !g_cudaDeviceApiFunctionTable.cuExternalMemoryGetMappedMipmappedArray
            || !g_cudaDeviceApiFunctionTable.cuDestroyExternalMemory
            || !g_cudaDeviceApiFunctionTable.cuImportExternalSemaphore
            || !g_cudaDeviceApiFunctionTable.cuSignalExternalSemaphoresAsync
            || !g_cudaDeviceApiFunctionTable.cuWaitExternalSemaphoresAsync
            || !g_cudaDeviceApiFunctionTable.cuDestroyExternalSemaphore
            || !g_cudaDeviceApiFunctionTable.cuModuleLoad
            || !g_cudaDeviceApiFunctionTable.cuModuleLoadData
            || !g_cudaDeviceApiFunctionTable.cuModuleLoadDataEx
            || !g_cudaDeviceApiFunctionTable.cuModuleLoadFatBinary
            || !g_cudaDeviceApiFunctionTable.cuModuleUnload
            || !g_cudaDeviceApiFunctionTable.cuModuleGetFunction
            || !g_cudaDeviceApiFunctionTable.cuModuleGetGlobal
            || !g_cudaDeviceApiFunctionTable.cuLaunchKernel
            || !g_cudaDeviceApiFunctionTable.cuOccupancyMaxPotentialBlockSize) {
        throw std::runtime_error(
                "Error in initializeCudaDeviceApiFunctionTable: "
                "At least one function pointer could not be loaded.");
    }

    return true;
}

#ifdef _WIN32
#undef dlsym
#endif

bool getIsCudaDeviceApiFunctionTableInitialized() {
    return g_cudaLibraryHandle != nullptr;
}

void freeCudaDeviceApiFunctionTable() {
    if (g_cudaLibraryHandle) {
#if defined(__linux__)
        dlclose(g_cudaLibraryHandle);
#elif defined(_WIN32)
        FreeLibrary(g_cudaLibraryHandle);
#endif
        g_cudaLibraryHandle = {};
    }
}



CudaDeviceCoresInfo getNumCudaCores(CUdevice cuDevice) {
    CudaDeviceCoresInfo info{};

    /*
     * Only use one thread block per shader multiprocessor (SM) to improve chance of fair scheduling.
     * See, e.g.: https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954%5B/
     */
    CUresult cuResult;
    int numMultiprocessors = 16;
    cuResult = g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &numMultiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
    checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    info.numMultiprocessors = uint32_t(numMultiprocessors);

    /*
     * Use more threads than warp size. Factor 4 seems to make sense at least for RTX 3090.
     * For more details see: https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
     * Or: https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
     * https://developer.nvidia.com/blog/inside-pascal/
     */
    int warpSize = 32;
    cuResult = g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice);
    checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    info.warpSize = uint32_t(warpSize);

    int major = 0;
    cuResult = g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    int minor = 0;
    cuResult = g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
    checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");

    // Use warp size * 4 as fallback for unknown architectures.
    int numCoresPerMultiprocessor = warpSize * 4;

    if (major == 2) {
        if (minor == 1) {
            numCoresPerMultiprocessor = 48;
        } else {
            numCoresPerMultiprocessor = 32;
        }
    } else if (major == 3) {
        numCoresPerMultiprocessor = 192;
    } else if (major == 5) {
        numCoresPerMultiprocessor = 128;
    } else if (major == 6) {
        if (minor == 0) {
            numCoresPerMultiprocessor = 64;
        } else {
            numCoresPerMultiprocessor = 128;
        }
    } else if (major == 7) {
        numCoresPerMultiprocessor = 64;
    } else if (major == 8) {
        if (minor == 0) {
            numCoresPerMultiprocessor = 64;
        } else {
            numCoresPerMultiprocessor = 128;
        }
    } else if (major == 9) {
        numCoresPerMultiprocessor = 128;
    }
    info.numCoresPerMultiprocessor = uint32_t(numCoresPerMultiprocessor);
    info.numCudaCoresTotal = info.numMultiprocessors * info.numCoresPerMultiprocessor;

    return info;
}



void _checkCUresult(CUresult cuResult, const char* text, const char* locationText) {
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString = nullptr;
        cuResult = g_cudaDeviceApiFunctionTable.cuGetErrorString(cuResult, &errorString);
        if (cuResult == CUDA_SUCCESS) {
            throw std::runtime_error(std::string() + locationText + ": " + text + errorString);
        } else {
            throw std::runtime_error(std::string() + locationText + ": " + "Error in cuGetErrorString.");
        }
    }
}

void _checkNvrtcResult(nvrtcResult result, const char* text, const char* locationText) {
    if (result != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string() + locationText + ": " + text + nvrtcGetErrorString(result));
    }
}
