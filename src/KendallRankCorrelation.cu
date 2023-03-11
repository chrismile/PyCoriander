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
 * - MAX_STACK_SIZE: uint32_t(ceil(log(MEMBER_COUNT))) + 1; 11 for 1000 ensemble members.
 */

// ----------------------------------------------------------------------------------
/*
 * SortJoint.
 */

__device__ void swapElementsJoint(float* referenceValues, float* queryValues, uint i, uint j) {
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
}

__device__ void heapifyJoint(float* referenceValues, float* queryValues, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = referenceValues[child];
        childValue1 = referenceValues[child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (referenceValues[i] >= childValue0) {
            break;
        }
        swapElementsJoint(referenceValues, queryValues, i, child);
        i = child;
    }
}

__device__ void heapSortJoint(float* referenceValues, float* queryValues) {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapifyJoint(referenceValues, queryValues, i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElementsJoint(referenceValues, queryValues, 0, MEMBER_COUNT - i);
        heapifyJoint(referenceValues, queryValues, 0, MEMBER_COUNT - i);
    }
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
/*
 * SortRange.
 */

__device__ void swapElementsRange(float* sortArray, uint i, uint j) {
    float temp = sortArray[i];
    sortArray[i] = sortArray[j];
    sortArray[j] = temp;
}

__device__ void heapifyRange(float* sortArray, uint startIdx, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1, arrayI;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = sortArray[startIdx + child];
        childValue1 = sortArray[startIdx + child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (sortArray[startIdx + i] >= childValue0) {
            break;
        }
        swapElementsRange(sortArray, startIdx + i, startIdx + child);
        i = child;
    }
}

__device__ void heapSortRange(float* sortArray, uvec2 range) {
    uint numElements = range.y - range.x + 1;

    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = numElements / 2; i > 0; i--) {
        heapifyRange(sortArray, range.x, i - 1, numElements);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < numElements; i++) {
        swapElementsRange(sortArray, range.x, range.x + numElements - i);
        heapifyRange(sortArray, range.x, 0, numElements - i);
    }
}
// ----------------------------------------------------------------------------------


// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
__device__ uint M(float* sortArray, uvec2 leftRange, uvec2 rightRange) {
    uint i = leftRange.x;
    uint j = rightRange.x;
    uint numSwaps = 0;
    while (i <= leftRange.y && j <= rightRange.y) {
        if (sortArray[j] < sortArray[i]) {
            numSwaps += leftRange.y + 1 - i;
            j += 1;
        } else {
            i += 1;
        }
    }
    return numSwaps;
}

__device__ uint S(float* queryValues, float* sortArray) {
    uint sum = 0;
    uvec2 stack[MAX_STACK_SIZE];
    stack[0] = make_uint2(0, MEMBER_COUNT - 1);
    uint stackSize = 1;
    while (stackSize > 0) {
        uvec2 range = stack[stackSize - 1];
        stackSize--;
        if (range.y - range.x == 0) {
            continue;
        }
        uint s = (range.y - range.x + 1) / 2;
        for (uint i = range.x; i <= range.y; i++) {
            sortArray[i] = queryValues[i];
        }
        uvec2 rangeLeft = make_uint2(range.x, range.x + s - 1);
        uvec2 rangeRight = make_uint2(range.x + s, range.y);
        heapSortRange(sortArray, rangeLeft);
        heapSortRange(sortArray, rangeRight);
        sum += M(sortArray, rangeLeft, rangeRight);
        stack[stackSize] = rangeLeft;
        stack[stackSize + 1] = rangeRight;
        stackSize += 2;
    }
    return sum;
}

__global__ void kendallRankCorrelation(
        const float* __restrict__ referenceArray, const float* __restrict__ queryArray, float* __restrict__ outputArray,
        const uint32_t referenceStride, const uint32_t queryStride,
        const uint32_t batchOffset, const uint32_t batchSize) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    float referenceValues[MEMBER_COUNT];
    float queryValues[MEMBER_COUNT];
    float sortArray[MEMBER_COUNT];
    uint offsetReferenceValues = globalThreadIdx * referenceStride;
    uint offsetQueryValues = globalThreadIdx * queryStride;

    for (uint c = 0; c < MEMBER_COUNT; c++) {
        referenceValues[c] = referenceArray[offsetReferenceValues + c];
        queryValues[c] = queryArray[offsetQueryValues + c];
    }

    heapSortJoint(referenceValues, queryValues);
    int S_y = int(S(queryValues, sortArray));

    int n0 = (MEMBER_COUNT * (MEMBER_COUNT - 1)) / 2;

    // Use Tau-a statistic without accounting for ties for now.
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;

    int numerator = n0 - n1 - n2 + n3 - 2 * S_y;
    // The square root needs to be taken separately to avoid integer overflow.
    float denominator = sqrt(float(n0 - n1)) * sqrt(float(n0 - n2));
    float correlationValue = float(numerator) / denominator;

    outputArray[globalThreadIdx] = correlationValue;
}

}
