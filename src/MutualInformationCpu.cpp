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

#include "MutualInformation.hpp"
#include "MutualInformationCpu.hpp"

PYBIND11_MODULE(pycoriander, m) {
    m.def("_cleanup", pycorianderCleanup, "Cleanup correlation estimator data");
    m.def("mutual_information_kraskov", mutualInformationKraskov, "Mutual information using Kraskov estimator");
}

torch::Tensor mutualInformationKraskov(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k) {
    if (referenceTensor.device().is_cpu()) {
        return mutualInformationKraskovCpu(referenceTensor, queryTensor, k);
    } else if (referenceTensor.device().is_cuda()) {
        return mutualInformationKraskovCuda(referenceTensor, queryTensor, k);
    } else {
        throw std::runtime_error("Error in mutualInformationKraskov: Unsupported device.");
    }
}

torch::Tensor mutualInformationKraskovCpu(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k) {
    if (referenceTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in mutualInformationKraskovCpu: referenceTensor.sizes().size() > 1.");
    }
    if (queryTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in mutualInformationKraskovCpu: queryTensor.sizes().size() > 2.");
    }

    // Size of tensor: (M, N) or (N).
    const int64_t Mr = referenceTensor.sizes().size() == 1 ? 1 : referenceTensor.size(0);
    const int64_t Nr = referenceTensor.sizes().size() == 1 ? referenceTensor.size(0) : referenceTensor.size(1);
    const int64_t Mq = queryTensor.sizes().size() == 1 ? 1 : queryTensor.size(0);
    const int64_t Nq = queryTensor.sizes().size() == 1 ? queryTensor.size(0) : queryTensor.size(1);
    if (Nr != Nq || (Mr != Mq && Mr != 1 && Mq != 1)) {
        throw std::runtime_error("Error in mutualInformationKraskovCuda: Tensor size mismatch.");
    }
    const int64_t M = std::max(Mr, Mq);
    const int64_t N = Nr;

    torch::Tensor outputTensor = torch::zeros(M, at::TensorOptions().dtype(torch::kFloat32));
    auto referenceData = referenceTensor.data_ptr<float>();
    auto queryData = queryTensor.data_ptr<float>();
    auto outputAccessor = outputTensor.accessor<float, 1>();
    auto referenceStride = referenceTensor.sizes().size() == 1 ? 0 : uint32_t(referenceTensor.stride(0));
    auto queryStride = queryTensor.sizes().size() == 1 ? 0 : uint32_t(queryTensor.stride(0));

#ifdef _OPENMP
    #pragma omp parallel default(none) shared(M, N, k, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
            float* referenceValues = referenceData + batchIdx * referenceStride;
            float* queryValues = queryData + batchIdx * queryStride;
            float miValue = computeMutualInformationKraskov<float>(referenceValues, queryValues, int(k), int(N));
            outputAccessor[batchIdx] = miValue;
        }
    }

    return outputTensor;
}
