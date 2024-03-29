/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020-2023, Christoph Neuhauser
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

#ifndef PYCORIANDER_CUDAHELPERS_HPP
#define PYCORIANDER_CUDAHELPERS_HPP

#include <iostream>
#include <unordered_map>

#include <cuda.h>
#include <nvrtc.h>

struct CudaDeviceApiFunctionTable {
    CUresult ( *cuInit )( unsigned int Flags );
    CUresult ( *cuGetErrorString )( CUresult error, const char** pStr );

    CUresult ( *cuDeviceGet )(CUdevice* device, int ordinal);
    CUresult ( *cuDeviceGetCount )(int* count);
    CUresult ( *cuDeviceGetUuid )(CUuuid* uuid, CUdevice dev);
    CUresult ( *cuDeviceGetAttribute )(int* pi, CUdevice_attribute attrib, CUdevice dev);

    CUresult ( *cuCtxCreate )( CUcontext* pctx, unsigned int flags, CUdevice dev );
    CUresult ( *cuCtxDestroy )( CUcontext ctx );

    CUresult ( *cuStreamCreate )( CUstream* phStream, unsigned int Flags );
    CUresult ( *cuStreamDestroy )( CUstream hStream );
    CUresult ( *cuStreamSynchronize )( CUstream hStream );

    CUresult ( *cuMemAlloc )( CUdeviceptr* dptr, size_t bytesize );
    CUresult ( *cuMemFree )( CUdeviceptr dptr );
    CUresult ( *cuMemcpyDtoH )( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount );
    CUresult ( *cuMemcpyHtoD )( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );
    CUresult ( *cuMemAllocAsync )( CUdeviceptr* dptr, size_t bytesize, CUstream hStream );
    CUresult ( *cuMemFreeAsync )( CUdeviceptr dptr, CUstream hStream );
    CUresult ( *cuMemsetD8Async )( CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream );
    CUresult ( *cuMemsetD16Async )( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream );
    CUresult ( *cuMemsetD32Async )( CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream );
    CUresult ( *cuMemcpyAsync )( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream );
    CUresult ( *cuMemcpyDtoHAsync )( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
    CUresult ( *cuMemcpyHtoDAsync )( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream );
    CUresult ( *cuMemcpy2DAsync )( const CUDA_MEMCPY2D* pCopy, CUstream hStream );
    CUresult ( *cuMemcpy3DAsync )( const CUDA_MEMCPY3D* pCopy, CUstream hStream );

    CUresult ( *cuArrayCreate )( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray );
    CUresult ( *cuArray3DCreate )( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray );
    CUresult ( *cuArrayDestroy )( CUarray hArray );
    CUresult ( *cuMipmappedArrayCreate )( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels );
    CUresult ( *cuMipmappedArrayDestroy )( CUmipmappedArray hMipmappedArray );
    CUresult ( *cuMipmappedArrayGetLevel )( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level );

    CUresult ( *cuTexObjectCreate )( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc );
    CUresult ( *cuTexObjectDestroy )( CUtexObject texObject );
    CUresult ( *cuSurfObjectCreate )( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc );
    CUresult ( *cuSurfObjectDestroy )( CUsurfObject surfObject );

    CUresult ( *cuImportExternalMemory )( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc );
    CUresult ( *cuExternalMemoryGetMappedBuffer )( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc );
    CUresult ( *cuExternalMemoryGetMappedMipmappedArray )( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc );
    CUresult ( *cuDestroyExternalMemory )( CUexternalMemory extMem );

    CUresult ( *cuImportExternalSemaphore )( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc );
    CUresult ( *cuSignalExternalSemaphoresAsync )( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream );
    CUresult ( *cuWaitExternalSemaphoresAsync )( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream );
    CUresult ( *cuDestroyExternalSemaphore )( CUexternalSemaphore extSem );

    CUresult ( *cuModuleLoad )( CUmodule* module, const char* fname );
    CUresult ( *cuModuleLoadData )( CUmodule* module, const void* image );
    CUresult ( *cuModuleLoadDataEx )( CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues );
    CUresult ( *cuModuleLoadFatBinary )( CUmodule* module, const void* fatCubin );
    CUresult ( *cuModuleUnload )( CUmodule hmod );
    CUresult ( *cuModuleGetFunction )( CUfunction* hfunc, CUmodule hmod, const char* name );
    CUresult ( *cuModuleGetGlobal )( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name );
    CUresult ( *cuLaunchKernel )( CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra );
    CUresult ( *cuOccupancyMaxPotentialBlockSize )( int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit );
};

extern CudaDeviceApiFunctionTable g_cudaDeviceApiFunctionTable;

bool initializeCudaDeviceApiFunctionTable();
bool getIsCudaDeviceApiFunctionTableInitialized();
void freeCudaDeviceApiFunctionTable();


struct CudaDeviceCoresInfo {
    uint32_t numMultiprocessors;
    uint32_t warpSize;
    uint32_t numCoresPerMultiprocessor;
    uint32_t numCudaCoresTotal;
};
CudaDeviceCoresInfo getNumCudaCores(CUdevice cuDevice);



#ifndef TOSTRING
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#endif

void _checkNvrtcResult(nvrtcResult result, const char* text, const char* locationText);
#define checkNvrtcResult(result, text) _checkNvrtcResult(result, text, __FILE__ ":" TOSTRING(__LINE__))

void _checkCUresult(CUresult cuResult, const char* text, const char* locationText);
#define checkCUresult(cuResult, text) _checkCUresult(cuResult, text, __FILE__ ":" TOSTRING(__LINE__))



// Integer ceiling operation, i.e., ceil(x/y)
inline int iceil(int x, int y) { return 1 + ((x - 1) / y); }
// Avoids going into negative for x == 0 and overflow.
inline uint32_t uiceil(uint32_t x, uint32_t y) { return x > 0 ? (x - 1) / y + 1 : 0; }

// Integer binary logarithm.
inline int ilog2(int x) {
    int log2x = 0;
    while ((x >>= 1) != 0) {
        ++log2x;
    }
    return log2x;
}

inline bool isPowerOfTwo(int x) { return (x != 0) && ((x & (x - 1)) == 0); }
inline int nextPowerOfTwo(int x) { --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return x+1; }
inline int lastPowerOfTwo(int x) { return nextPowerOfTwo(x/2+1); }

#endif //PYCORIANDER_CUDAHELPERS_HPP
