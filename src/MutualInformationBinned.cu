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
typedef float2 vec2;

#define FLT_MAX 1e18f

/*
 * Global defines:
 * - MEMBER_COUNT: Number of entries to compute the correlation for.
 * - numBins: The number of bins to use for discretization.
 */

__device__ inline int clamp(int x, int a, int b) {
    return x <= a ? a : (x >= b ? b : x);
}

__global__ void mutualInformationBinned(
        const float* __restrict__ referenceArray, const float* __restrict__ queryArray, float* __restrict__ outputArray,
        const uint32_t referenceStride, const uint32_t queryStride,
        const uint32_t batchOffset, const uint32_t batchSize,
        const float referenceMin, const float referenceMax, const float queryMin, const float queryMax) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    uint offsetReferenceValues = globalThreadIdx * referenceStride;
    uint offsetQueryValues = globalThreadIdx * queryStride;

    float histogram2d[numBins * numBins];
    float histogram0[numBins];
    float histogram1[numBins];

    // Initialize the histograms with zeros.
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        histogram0[binIdx] = 0;
        histogram1[binIdx] = 0;
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] = 0;
        }
    }

    // Compute the 2D joint histogram.
    for (int c = 0; c < MEMBER_COUNT; c++) {
        float val0 = referenceArray[offsetReferenceValues + c];
        float val1 = queryArray[offsetQueryValues + c];
        val0 = (val0 - referenceMin) / (referenceMax - referenceMin);
        val1 = (val1 - queryMin) / (queryMax - queryMin);
        int binIdx0 = clamp(int(val0 * float(numBins)), 0, numBins - 1);
        int binIdx1 = clamp(int(val1 * float(numBins)), 0, numBins - 1);
        histogram2d[binIdx0 * numBins + binIdx1] += 1;
    }

    // Normalize the histograms.
    float totalSum2d = 0.0;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            totalSum2d += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] /= totalSum2d;
        }
    }

    // Marginalization of joint distribution.
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram0[binIdx0] += histogram2d[binIdx0 * numBins + binIdx1];
            histogram1[binIdx1] += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }

    /*
     * Compute the mutual information metric. Two possible ways of calculation:
     * a) $MI = H(x) + H(y) - H(x, y)$
     * with the Shannon entropy $H(x) = -\sum_i p_x(i) \log p_x(i)$
     * and the joint entropy $H(x, y) = -\sum_i \sum_j p_{xy}(i, j) \log p_{xy}(i, j)$
     * b) $MI = \sum_i \sum_j p_{xy}(i, j) \log \frac{p_{xy}(i, j)}{p_x(i) p_y(j)}$
     */
    const float EPSILON_1D = 0.5 / float(MEMBER_COUNT);
    const float EPSILON_2D = 0.5 / float(MEMBER_COUNT * MEMBER_COUNT);
    float mi = 0.0;
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        float p_x = histogram0[binIdx];
        float p_y = histogram1[binIdx];
        if (p_x > EPSILON_1D) {
            mi -= p_x * log(p_x);
        }
        if (p_y > EPSILON_1D) {
            mi -= p_y * log(p_y);
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            float p_xy = histogram2d[binIdx0 * numBins + binIdx1];
            if (p_xy > EPSILON_2D) {
                mi += p_xy * log(p_xy);
            }
        }
    }

    outputArray[globalThreadIdx] = mi;
}

}
