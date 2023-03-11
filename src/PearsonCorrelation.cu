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

extern "C" {

typedef unsigned uint;
typedef unsigned uint32_t;
typedef uint2 uvec2;

/*
 * Global defines:
 * - MEMBER_COUNT: Number of entries to compute the correlation for.
 */

__global__ void pearsonCorrelation(
        const float* __restrict__ referenceArray, const float* __restrict__ queryArray, float* __restrict__ outputArray,
        const uint32_t referenceStride, const uint32_t queryStride,
        const uint32_t batchOffset, const uint32_t batchSize) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    uint offsetReferenceValues = globalThreadIdx * referenceStride;
    uint offsetQueryValues = globalThreadIdx * queryStride;

    float n = float(MEMBER_COUNT);
    float meanX = 0;
    float meanY = 0;
    float invN = float(1) / n;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceArray[offsetReferenceValues + c];
        float y = queryArray[offsetQueryValues + c];
        meanX += invN * x;
        meanY += invN * y;
    }
    float varX = 0;
    float varY = 0;
    float invNm1 = float(1) / (n - float(1));
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceArray[offsetReferenceValues + c];
        float y = queryArray[offsetQueryValues + c];
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    float pearsonCorrelation = 0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceArray[offsetReferenceValues + c];
        float y = queryArray[offsetQueryValues + c];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }

    outputArray[globalThreadIdx] = pearsonCorrelation;
}

}
