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

__device__ void swapElements(float* valueArray, uint* ordinalRankArray, uint i, uint j) {
    float ftemp = valueArray[i];
    valueArray[i] = valueArray[j];
    valueArray[j] = ftemp;
    uint utemp = ordinalRankArray[i];
    ordinalRankArray[i] = ordinalRankArray[j];
    ordinalRankArray[j] = utemp;
}

__device__ void heapify(float* valueArray, uint* ordinalRankArray, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = valueArray[child];
        childValue1 = valueArray[child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (valueArray[i] >= childValue0) {
            break;
        }
        swapElements(valueArray, ordinalRankArray, i, child);
        i = child;
    }
}

__device__ void heapSort(float* valueArray, uint* ordinalRankArray) {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapify(valueArray, ordinalRankArray, i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElements(valueArray, ordinalRankArray, 0, MEMBER_COUNT - i);
        heapify(valueArray, ordinalRankArray, 0, MEMBER_COUNT - i);
    }
}

__device__ void computeFractionalRanking(const float* valueArray, const uint* ordinalRankArray, float* rankArray) {
    float currentRank = 1.0f;
    int idx = 0;
    while (idx < MEMBER_COUNT) {
        float value = valueArray[idx];
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < MEMBER_COUNT && value == valueArray[idxEqualEnd]) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        float meanRank = currentRank + float(numEqualValues - 1) * 0.5f;
        for (int offset = 0; offset < numEqualValues; offset++) {
            rankArray[ordinalRankArray[idx + offset]] = meanRank;
        }

        idx += numEqualValues;
        currentRank += float(numEqualValues);
    }
}

__device__ float pearsonCorrelation(float* referenceRankArray, float* queryRankArray) {
    float n = float(MEMBER_COUNT);
    float meanX = 0;
    float meanY = 0;
    float invN = float(1) / n;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceRankArray[c];
        float y = queryRankArray[c];
        meanX += invN * x;
        meanY += invN * y;
    }
    float varX = 0;
    float varY = 0;
    float invNm1 = float(1) / (n - float(1));
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceRankArray[c];
        float y = queryRankArray[c];
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    float pearsonCorrelation = 0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceRankArray[c];
        float y = queryRankArray[c];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return pearsonCorrelation;
}

__global__ void spearmanRankCorrelation(
        const float* __restrict__ referenceArray, const float* __restrict__ queryArray, float* __restrict__ outputArray,
        const uint32_t referenceStride, const uint32_t queryStride,
        const uint32_t batchOffset, const uint32_t batchSize) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    float referenceRankArray[MEMBER_COUNT];
    float queryRankArray[MEMBER_COUNT];
    {
        float valueArray[MEMBER_COUNT];
        uint ordinalRankArray[MEMBER_COUNT];

        uint offsetReferenceValues = globalThreadIdx * referenceStride;
        // 1. Fill the value array.
        for (uint c = 0; c < MEMBER_COUNT; c++) {
            valueArray[c] = referenceArray[offsetReferenceValues + c];
            ordinalRankArray[c] = c;
        }
        // 2. Sort both arrays.
        heapSort(valueArray, ordinalRankArray);
        // 3. Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
        computeFractionalRanking(valueArray, ordinalRankArray, referenceRankArray);

        uint offsetQueryValues = globalThreadIdx * queryStride;
        for (uint c = 0; c < MEMBER_COUNT; c++) {
            valueArray[c] = queryArray[offsetQueryValues + c];
            ordinalRankArray[c] = c;
        }
        heapSort(valueArray, ordinalRankArray);
        computeFractionalRanking(valueArray, ordinalRankArray, queryRankArray);
    }

    // 4. Compute the Pearson correlation of the ranks.
    float correlation = pearsonCorrelation(referenceRankArray, queryRankArray);

    outputArray[globalThreadIdx] = correlation;
}

}
