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

#include "Correlation.hpp"
#include "MutualInformation.hpp"
#include "PyCoriander.hpp"

PYBIND11_MODULE(pycoriander, m) {
    m.def("_cleanup", pycorianderCleanup, "Cleanup correlation estimator data.");
    m.def("pearson_correlation", pearsonCorrelation,
          "Computes the Pearson correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("spearman_rank_correlation", spearmanRankCorrelation,
          "Computes the Spearman rank correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("kendall_rank_correlation", kendallRankCorrelation,
          "Computes the Kendall rank correlation coefficient of the Torch tensors X and Y.",
          py::arg("X"), py::arg("Y"));
    m.def("mutual_information_binned", mutualInformationBinned,
          "Computes the mutual information of the Torch tensors X and Y using a binning estimator.",
          py::arg("X"), py::arg("Y"), py::arg("num_bins"),
          py::arg("X_min"), py::arg("X_max"), py::arg("Y_min"), py::arg("Y_max"));
    m.def("mutual_information_kraskov", mutualInformationKraskov,
          "Computes the mutual information of the Torch tensors X and Y using the Kraskov estimator.",
          py::arg("X"), py::arg("Y"), py::arg("k"));
}

torch::Tensor pearsonCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::PEARSON, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else if (referenceTensor.device().is_cuda()) {
        return computeCorrelationCuda(
                referenceTensor, queryTensor, CorrelationMeasureType::PEARSON, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in pearsonCorrelation: Unsupported device.");
    }
}

torch::Tensor spearmanRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::SPEARMAN, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else if (referenceTensor.device().is_cuda()) {
        return computeCorrelationCuda(
                referenceTensor, queryTensor, CorrelationMeasureType::SPEARMAN, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in spearmanRankCorrelation: Unsupported device.");
    }
}

torch::Tensor kendallRankCorrelation(torch::Tensor referenceTensor, torch::Tensor queryTensor) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::KENDALL, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else if (referenceTensor.device().is_cuda()) {
        return computeCorrelationCuda(
                referenceTensor, queryTensor, CorrelationMeasureType::KENDALL, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in kendallRankCorrelation: Unsupported device.");
    }
}

torch::Tensor mutualInformationBinned(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t numBins,
        double referenceMin, double referenceMax, double queryMin, double queryMax) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_BINNED, int(numBins), 0,
                float(referenceMin), float(referenceMax), float(queryMin), float(queryMax));
    } else if (referenceTensor.device().is_cuda()) {
        return computeCorrelationCuda(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_BINNED, int(numBins), 0,
                float(referenceMin), float(referenceMax), float(queryMin), float(queryMax));
    } else {
        throw std::runtime_error("Error in mutualInformationBinned: Unsupported device.");
    }
}

torch::Tensor mutualInformationKraskov(torch::Tensor referenceTensor, torch::Tensor queryTensor, int64_t k) {
    if (referenceTensor.device().is_cpu()) {
        return computeCorrelationCpu(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV, 0, int(k),
                0.0f, 0.0f, 0.0f, 0.0f);
    } else if (referenceTensor.device().is_cuda()) {
        return computeCorrelationCuda(
                referenceTensor, queryTensor, CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV, 0, int(k),
                0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        throw std::runtime_error("Error in mutualInformationKraskov: Unsupported device.");
    }
}

torch::Tensor computeCorrelationCpu(
        torch::Tensor referenceTensor, torch::Tensor queryTensor, CorrelationMeasureType correlationMeasureType,
        int numBins, int k, float referenceMin, float referenceMax, float queryMin, float queryMax) {
    if (referenceTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in computeCorrelationCpu: referenceTensor.sizes().size() > 2.");
    }
    if (queryTensor.sizes().size() > 2) {
        throw std::runtime_error("Error in computeCorrelationCpu: queryTensor.sizes().size() > 2.");
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
    auto referenceStride = Mr == 1 ? 0 : uint32_t(referenceTensor.stride(0));
    auto queryStride = Mq == 1 ? 0 : uint32_t(queryTensor.stride(0));

    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computePearson2<float>(referenceValues, queryValues, int(N));
                outputAccessor[batchIdx] = miValue;
            }
        }
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            std::vector<std::pair<float, int>> ordinalRankArraySpearman;
            ordinalRankArraySpearman.reserve(N);
            auto* referenceRanks = new float[N];
            auto* queryRanks = new float[N];
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                computeRanks(referenceValues, referenceRanks, ordinalRankArraySpearman, int(N));
                computeRanks(queryValues, queryRanks, ordinalRankArraySpearman, int(N));
                float miValue = computePearson2<float>(referenceRanks, queryRanks, int(N));
                outputAccessor[batchIdx] = miValue;
            }
            delete[] referenceRanks;
            delete[] queryRanks;
        }
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(N);
            ordinalRankArray.reserve(N);
            y.reserve(N);
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computeKendall(referenceValues, queryValues, int(N), jointArray, ordinalRankArray, y);
                outputAccessor[batchIdx] = miValue;
            }
        }
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, numBins, referenceData, referenceStride, queryData, queryStride, outputAccessor) \
        shared(referenceMin, referenceMax, queryMin, queryMax)
#endif
        {
            auto* histogram0 = new float[numBins];
            auto* histogram1 = new float[numBins];
            auto* histogram2d = new float[numBins * numBins];
            auto* X = new float[N];
            auto* Y = new float[N];
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                for (int i = 0; i < N; i++) {
                    X[i] = (referenceValues[i] - referenceMin) / (referenceMax - referenceMin);
                    Y[i] = (queryValues[i] - queryMin) / (queryMax - queryMin);
                }
                float miValue = computeMutualInformationBinned<float>(
                        X, Y, int(numBins), int(N), histogram0, histogram1, histogram2d);
                outputAccessor[batchIdx] = miValue;
            }
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
            delete[] X;
            delete[] Y;
        }
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
#ifdef _OPENMP
        #pragma omp parallel default(none) shared(M, N, k, referenceData, referenceStride, queryData, queryStride, outputAccessor)
#endif
        {
            KraskovEstimatorCache<float> kraskovEstimatorCache;
#ifdef _OPENMP
            #pragma omp for
#endif
            for (int batchIdx = 0; batchIdx < int(M); batchIdx++) {
                float* referenceValues = referenceData + batchIdx * referenceStride;
                float* queryValues = queryData + batchIdx * queryStride;
                float miValue = computeMutualInformationKraskov<float>(
                        referenceValues, queryValues, int(k), int(N), kraskovEstimatorCache);
                outputAccessor[batchIdx] = miValue;
            }
        }
    }

    return outputTensor;
}
