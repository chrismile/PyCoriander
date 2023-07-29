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

extern "C" {

typedef unsigned uint;
typedef unsigned uint32_t;
typedef uint3 uvec3;
typedef float2 vec2;

/*
 * For more details, please refer to:
 * - https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
 *
 * Global defines:
 * - MEMBER_COUNT: Number of entries to compute the MI for.
 * - k: Number of neighbors used in search.
 * - MAX_STACK_SIZE_BUILD: 2 * uint32_t(ceil(log(MEMBER_COUNT + 1))); 2*10 for 1000 entries.
 * - MAX_STACK_SIZE_KN: uint32_t(ceil(log(MEMBER_COUNT + 1))); 10 for 1000 entries.
 */

const float EPSILON = 1e-6;
const float EPSILON_NOISE = 1e-5;

#define KRASKOV_USE_RANDOM_NOISE

#ifdef KRASKOV_USE_RANDOM_NOISE
__device__ float getRandomFloatNorm(uvec3& rngState) {
    rngState.x ^= rngState.x << 16;
    rngState.x ^= rngState.x >> 5;
    rngState.x ^= rngState.x << 1;

    uint t = rngState.x;
    rngState.x = rngState.y;
    rngState.y = rngState.z;
    rngState.z = t ^ rngState.x ^ rngState.y;

    return rngState.z / float(4294967295u) * 2.0 - 1.0;
}
#endif


/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 *
 * This function could be extended for values < 1 by:
 * - float z = 1 - iz;
 * - if (iz < 1) return digammaValue - M_PI * cosf(M_PI * iz) / sinf(M_PI * iz);
 */
#define G (5.15f)
#define P0 (2.50662827563479526904f)
#define P1 (225.525584619175212544f)
#define P2 (-268.295973841304927459f)
#define P3 (80.9030806934622512966f)
#define P4 (-5.00757863970517583837f)
#define P5 (0.0114684895434781459556f)
__device__ float digamma(uint iz) {
    if (iz == 1u) {
        return -0.57721566490153287f;
    }
    float z = float(iz);
    float zh = z - 0.5f;
    float z1 = z + 1.0f;
    float z2 = z + 2.0f;
    float z3 = z + 3.0f;
    float z4 = z + 4.0f;
    float ZP = P0 + P1 / z + P2 / z1 + P3 / z2 + P4 / z3 + P5 / z4;
    float dZP = P1 / (z * z) + P2 / (z1 * z1) + P3 / (z2 * z2) + P4 / (z3 * z3) + P5 / (z4 * z4);
    float digammaValue = logf(zh + G) + zh / (zh + G) - dZP / ZP - 1.0f;
    return digammaValue;
}


// ----------------------------------------------------------------------------------
/*
 * Sort2D.
 */
__device__ void swapElements2D(float* referenceValues, float* queryValues, uint i, uint j) {
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
}

__device__ void heapify2D(
        float* referenceValues, float* queryValues, float* valuesAxis,
        uint startIdx, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1, arrayI;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = valuesAxis[startIdx + child];
        childValue1 = valuesAxis[startIdx + child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        arrayI = valuesAxis[startIdx + i];
        if (arrayI >= childValue0) {
            break;
        }
        swapElements2D(referenceValues, queryValues, startIdx + i, startIdx + child);
        i = child;
    }
}

__device__ void heapSort2D(
        float* referenceValues, float* queryValues, float* valuesAxis,
        uint startIdx, uint endIdx) {
    uint numElements = endIdx - startIdx;

    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = numElements / 2; i > 0; i--) {
        heapify2D(referenceValues, queryValues, valuesAxis, startIdx, i - 1, numElements);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < numElements; i++) {
        swapElements2D(referenceValues, queryValues, startIdx, startIdx + numElements - i);
        heapify2D(referenceValues, queryValues, valuesAxis, startIdx, 0, numElements - i);
    }
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
/*
 * Sort1D.
 */
__device__ void swapElements(
        float* kthNeighborDistances, float* valueArray, uint i, uint j) {
    float temp = valueArray[i];
    valueArray[i] = valueArray[j];
    valueArray[j] = temp;
    temp = kthNeighborDistances[i];
    kthNeighborDistances[i] = kthNeighborDistances[j];
    kthNeighborDistances[j] = temp;
}

__device__ void heapify(
        float* kthNeighborDistances, float* valueArray,
        uint i, uint numElements) {
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
        swapElements(kthNeighborDistances, valueArray, i, child);
        i = child;
    }
}

__device__ void heapSort(float* kthNeighborDistances, float* valueArray) {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapify(kthNeighborDistances, valueArray, i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElements(kthNeighborDistances, valueArray, 0, MEMBER_COUNT - i);
        heapify(kthNeighborDistances, valueArray, 0, MEMBER_COUNT - i);
    }
}

__device__ float averageDigamma(float* kthNeighborDistances, float* valueArray) {
    heapSort(kthNeighborDistances, valueArray);
    float factor = 1.0 / float(MEMBER_COUNT);
    float meanDigammaValue = 0.0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float kthDist = kthNeighborDistances[c] - EPSILON;
        float currentValue = valueArray[c];
        float searchValueLower = currentValue - kthDist;
        float searchValueUpper = currentValue + kthDist;
        int lower = 0;
        int upper = MEMBER_COUNT;
        int middle = 0;
        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            float middleValue = valueArray[middle];
            if (middleValue < searchValueLower) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        int startRange = upper;
        lower = startRange;
        upper = MEMBER_COUNT;

        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            float middleValue = valueArray[middle];
            if (middleValue < searchValueUpper) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        int endRange = upper - 1;

        uint numPoints = max(uint(endRange + 1 - startRange), 1u);
        meanDigammaValue += factor * digamma(numPoints);
    }
    return meanDigammaValue;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
/*
 * k-d tree.
 */
const uint INVALID_NODE = 0xFFFFu;
#define FLT_MAX 3.402823466e+38

struct KdNode {
    __device__ KdNode() {}
    __device__ KdNode(float x, float y, uint axis, uint leftRightIdx) : axis(axis), leftRightIdx(leftRightIdx) {
        this->point[0] = x;
        this->point[1] = y;
    }
    float point[2];
    uint axis;
    uint leftRightIdx;
};

struct StackEntryBuild {
    __device__ StackEntryBuild() {}
    __device__ StackEntryBuild(uint startIdx, uint endIdx, uint depth) : startIdx(startIdx), endIdx(endIdx), depth(depth) {}
    uint startIdx;
    uint endIdx;
    uint depth;
};

__device__ void buildKdTree(KdNode* nodes, float* referenceValues, float* queryValues) {
    uint nodeCounter = 0;
    StackEntryBuild stack[MAX_STACK_SIZE_BUILD];
    uint stackSize = 1u;
    stack[0] = StackEntryBuild(0u, MEMBER_COUNT, 0u);
    StackEntryBuild stackEntry;
    while (stackSize > 0u) {
        stackSize--;
        stackEntry = stack[stackSize];

        uint axis = stackEntry.depth % 2u;
        heapSort2D(
                referenceValues, queryValues, axis == 0u ? referenceValues : queryValues,
                stackEntry.startIdx, stackEntry.endIdx);
        uint medianIndex = stackEntry.startIdx + (stackEntry.endIdx - stackEntry.startIdx) / 2u;

        uint rightIdx;
        if (stackEntry.endIdx - medianIndex - 1 == 0u) {
            rightIdx = INVALID_NODE;
        } else {
            rightIdx = nodeCounter + medianIndex - stackEntry.startIdx + 1;
            stack[stackSize] = StackEntryBuild(medianIndex + 1, stackEntry.endIdx, stackEntry.depth + 1);
            stackSize++;
        }

        uint leftIdx;
        if (medianIndex - stackEntry.startIdx == 0u) {
            leftIdx = INVALID_NODE;
        } else {
            leftIdx = nodeCounter + 1;
            stack[stackSize] = StackEntryBuild(stackEntry.startIdx, medianIndex, stackEntry.depth + 1);
            stackSize++;
        }

        uint leftRightIdx = (leftIdx | rightIdx << 16u);
        nodes[nodeCounter] = KdNode(referenceValues[medianIndex], queryValues[medianIndex], axis, leftRightIdx);
        nodeCounter++;
    }
}

__device__ float findKNearestNeighbors(KdNode* nodes, float point[2], uint e) {
    float distances[k + 1];
    #pragma unroll
    for (int i = 0; i <= k; i++) {
        distances[i] = FLT_MAX;
    }

    uint stack[MAX_STACK_SIZE_KN];
    uint stackSize = 0u;
    uint currNodeIdx = 0u;
    KdNode currNode;
    while (currNodeIdx != INVALID_NODE || stackSize > 0u) {
        while (currNodeIdx != INVALID_NODE) {
            stack[stackSize] = currNodeIdx;
            stackSize++;
            currNode = nodes[currNodeIdx];

            // Descend on side of split planes where the point lies.
            bool isPointOnLeftSide = point[currNode.axis] <= currNode.point[currNode.axis];
            if (isPointOnLeftSide) {
                currNodeIdx = currNode.leftRightIdx & 0x0000FFFFu;
            } else {
                currNodeIdx = (currNode.leftRightIdx & 0xFFFF0000u) >> 16u;
            }
        }

        stackSize--;
        currNodeIdx = stack[stackSize];
        currNode = nodes[currNodeIdx];

        // Compute the distance of this node to the point.
        vec2 diff = make_float2(fabs(point[0] - currNode.point[0]), fabs(point[1] - currNode.point[1]));
        float newDistance = max(diff.x, diff.y);
        if (newDistance < distances[k]) {
            float tempDistance;
            for (int i = 0; i <= k; i++) {
                if (newDistance < distances[i]) {
                    tempDistance = newDistance;
                    newDistance = distances[i];
                    distances[i] = tempDistance;
                }
            }
        }

        // Check whether there could be a closer point on the opposite side.
        bool isPointOnLeftSide = point[currNode.axis] <= currNode.point[currNode.axis];
        if (isPointOnLeftSide && point[currNode.axis] + distances[k] >= currNode.point[currNode.axis]) {
            currNodeIdx = (currNode.leftRightIdx & 0xFFFF0000u) >> 16u;
        } else if (!isPointOnLeftSide && point[currNode.axis] - distances[k] <= currNode.point[currNode.axis]) {
            currNodeIdx = currNode.leftRightIdx & 0x0000FFFFu;
        } else {
            currNodeIdx = INVALID_NODE;
        }
    }
    return distances[k];
}
// ----------------------------------------------------------------------------------


__global__ void mutualInformationKraskov(
        const float* __restrict__ referenceArray, const float* __restrict__ queryArray, float* __restrict__ miArray,
        const uint32_t referenceStride, const uint32_t queryStride,
        const uint32_t batchOffset, const uint32_t batchSize) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    float referenceValues[MEMBER_COUNT];
    float queryValues[MEMBER_COUNT];
    uint offsetReferenceValues = globalThreadIdx * referenceStride;
    uint offsetQueryValues = globalThreadIdx * queryStride;

#ifdef KRASKOV_USE_RANDOM_NOISE
    uint seed = 17u * globalThreadIdx + 240167u;

    // Use Xorshift random numbers with period 2^96-1.
    uvec3 rngState;
    rngState.x = 123456789u ^ seed;
    rngState.y = 362436069u ^ seed;
    rngState.z = 521288629u ^ seed;

    // Optionally add noise.
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        referenceValues[c] = referenceArray[offsetReferenceValues + c] + EPSILON_NOISE * getRandomFloatNorm(rngState);
        queryValues[c] = queryArray[offsetQueryValues + c] + EPSILON_NOISE * getRandomFloatNorm(rngState);
    }
#else
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        referenceValues[c] = referenceArray[offsetReferenceValues + c];
        queryValues[c] = queryArray[offsetQueryValues + c];
    }
#endif

    KdNode nodes[MEMBER_COUNT];
    buildKdTree(nodes, referenceValues, queryValues);

    float point[2];
    float kthNeighborDistances0[MEMBER_COUNT];
    float kthNeighborDistances1[MEMBER_COUNT];
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        point[0] = referenceValues[c];
        point[1] = queryValues[c];
        float val = findKNearestNeighbors(nodes, point, c);
        kthNeighborDistances0[c] = val;
        kthNeighborDistances1[c] = val;
    }

    float a = averageDigamma(kthNeighborDistances0, referenceValues);
    float b = averageDigamma(kthNeighborDistances1, queryValues);
    float c = digamma(k);
    float d = digamma(MEMBER_COUNT);
    //float mi = (-a - b + c + d) / log(base);
    float mi = -a - b + c + d;

    miArray[globalThreadIdx] = mi;
}

}
