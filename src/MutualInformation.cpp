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

#include <algorithm>
#include <cmath>

#ifdef USE_BOOST_DIGAMMA
#include <boost/math/special_functions/digamma.hpp>
using namespace boost::math;
#else
#include "digamma.hpp"
#endif
#include "KdTreed.hpp"
#include "Random/Xorshift.hpp"

#include "MutualInformation.hpp"

template<class Real>
float computeMutualInformationBinned(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        Real* histogram0, Real* histogram1, Real* histogram2d) {
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
    for (int idx = 0; idx < es; idx++) {
        Real val0 = referenceValues[idx];
        Real val1 = queryValues[idx];
        if (!std::isnan(val0) && !std::isnan(val1)) {
            int binIdx0 = std::clamp(int(val0 * Real(numBins)), 0, numBins - 1);
            int binIdx1 = std::clamp(int(val1 * Real(numBins)), 0, numBins - 1);
            histogram2d[binIdx0 * numBins + binIdx1] += 1;
        }
    }

    // Normalize the histograms.
    Real totalSum2d = 0;
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

    // Regularize.
    /*const Real REG_FACTOR = 1e-7;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] += REG_FACTOR;
        }
    }

    // Normalize again.
    totalSum2d = 0;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            totalSum2d += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] /= totalSum2d;
        }
    }*/

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
    const Real EPSILON_1D = Real(0.5) / Real(es);
    const Real EPSILON_2D = Real(0.5) / Real(es * es);
    Real mi = 0.0;
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        Real p_x = histogram0[binIdx];
        Real p_y = histogram1[binIdx];
        if (p_x > EPSILON_1D) {
            mi -= p_x * std::log(p_x);
        }
        if (p_y > EPSILON_1D) {
            mi -= p_y * std::log(p_y);
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            Real p_xy = histogram2d[binIdx0 * numBins + binIdx1];
            if (p_xy > EPSILON_2D) {
                mi += p_xy * std::log(p_xy);
            }
        }
    }

    return float(mi);
}

template
float computeMutualInformationBinned<float>(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        float* histogram0, float* histogram1, float* histogram2d);
template
float computeMutualInformationBinned<double>(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        double* histogram0, double* histogram1, double* histogram2d);



#define KRASKOV_USE_RANDOM_NOISE
#define USE_1D_BINARY_SEARCH

template <typename FloatType> struct default_epsilon {};
template <> struct default_epsilon<float> { static const float value; static const float noise; };
template <> struct default_epsilon<double> { static const double value; static const double noise; };
const float default_epsilon<float>::value = 1e-6f;
const double default_epsilon<double>::value = 1e-15;
const float default_epsilon<float>::noise = 1e-5f;
const double default_epsilon<double>::noise = 1e-10;

template<class Real, bool includeCenter = true>
Real averageDigamma(
        const float* values, int es, const std::vector<Real>& distanceVec, bool isRef,
        KraskovEstimatorCache<Real>& cache) {
#ifdef KRASKOV_USE_RANDOM_NOISE
    XorshiftRandomGenerator gen(isRef ? 617406168ul : 864730169ul);
    cache.baseArray.resize(es);
#endif
    Real factor = Real(1) / Real(es);
    Real meanDigammaValue = 0;
#ifdef USE_1D_BINARY_SEARCH
    cache.sortedArray.resize(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        cache.sortedArray.at(e) = values[e] + gen.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise;
        cache.baseArray.at(e) = cache.sortedArray.at(e);
#else
        cache.sortedArray.at(e) = values[e];
#endif
    }
    std::sort(cache.sortedArray.begin(), cache.sortedArray.end());

    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        Real currentValue = cache.baseArray[e];
#else
        Real currentValue = values[e];
#endif
        Real kthDist;
        if constexpr(includeCenter) {
            kthDist = distanceVec[e] - default_epsilon<Real>::value;
        } else {
            kthDist = distanceVec[e] + default_epsilon<Real>::value;
        }
        Real searchValueLower = currentValue - kthDist;
        Real searchValueUpper = currentValue + kthDist;
        int lower = 0;
        int upper = es;
        int middle = 0;
        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = cache.sortedArray[middle];
            if (middleValue < searchValueLower) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        int startRange = upper;
        lower = startRange;
        upper = es;

        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = cache.sortedArray[middle];
            if (middleValue < searchValueUpper) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        int endRange = upper - 1;

        int numPoints = endRange + 1 - startRange;
        if constexpr(includeCenter) {
            meanDigammaValue += factor * Real(digamma(numPoints)); // nx/y + 1
        } else {
            meanDigammaValue += factor * Real(digamma(numPoints - 1)); // nx/y
        }
    }
#else
    KdTreed<Real, 1, DistanceMeasure::CHEBYSHEV> kdTree1d;
    std::vector<math::vec<1, Real>> points;
    points.reserve(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        points.emplace_back(values[e] + gen.getRandomFloatBetween(-1.0f, 1.0f) * default_epsilon<Real>::noise);
#else
        points.emplace_back(values[e]);
#endif
    }
    kdTree1d.build(points);

    for (int e = 0; e < es; e++) {
        auto numPoints = int(kdTree1d.getNumPointsInSphere(points.at(e), distanceVec.at(e) - default_epsilon<Real>::value));
        meanDigammaValue += factor * Real(digamma(numPoints));
    }
#endif
    return meanDigammaValue;
}

/*template<class Real>
void findKNearestNeighborDistances1D(
        int es, const std::vector<Real>& valueArray, const std::vector<Real>& sortedArray, std::vector<Real>& distanceVec) {
    for (int e = 0; e < es; e++) {
        Real currentValue = valueArray[e];

        Real kthDist = distanceVec[e] - default_epsilon<Real>::value;
        Real searchValueLower = currentValue - kthDist;
        Real searchValueUpper = currentValue + kthDist;
        int lower = 0;
        int upper = es;
        int middle = 0;
        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = sortedArray[middle];
            if (middleValue < searchValueLower) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        int startRange = upper;
        lower = startRange;
        upper = es;

        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = sortedArray[middle];
            if (middleValue < searchValueUpper) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        int endRange = upper - 1;
    }
}*/

/**
 * For more details, please refer to:
 * - https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
 * - https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
 */
template<class Real>
float computeMutualInformationKraskov(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<Real>& cache) {
    //const int base = 2;
    cache.points.clear();
    cache.pointsCopy.clear();
    cache.kdTree2d.clear();
    cache.kthNeighborDistances.clear();
    cache.nearestNeighborDistances.clear();

#ifdef KRASKOV_USE_RANDOM_NOISE
    XorshiftRandomGenerator genRef(617406168ul);
    XorshiftRandomGenerator genQuery(864730169ul);
#endif

    cache.points.reserve(es);
    cache.pointsCopy.reserve(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        cache.points.emplace_back(
                referenceValues[e] + genRef.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise,
                queryValues[e] + genQuery.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise);
#else
        cache.points.emplace_back(referenceValues[e], queryValues[e]);
#endif
        cache.pointsCopy.push_back(cache.points.back());
    }
    cache.kdTree2d.buildInplace(cache.pointsCopy);

    cache.kthNeighborDistances.reserve(es);
    cache.nearestNeighborDistances.reserve(k + 1);
    for (int e = 0; e < es; e++) {
        cache.nearestNeighborDistances.clear();
        cache.kdTree2d.findKNearestNeighbors(cache.points.at(e), k + 1, cache.nearestNeighborDistances);
        cache.kthNeighborDistances.emplace_back(cache.nearestNeighborDistances.back());
    }

    auto a = averageDigamma<Real>(referenceValues, es, cache.kthNeighborDistances, true, cache);
    auto b = averageDigamma<Real>(queryValues, es, cache.kthNeighborDistances, false, cache);
    auto c = Real(digamma(k));
    auto d = Real(digamma(es));

    //Real mi = (-a - b + c + d) / Real(std::log(base));
    Real mi = -a - b + c + d;
    return float(mi);
}

/**
 * Second estimator by Kraskov et al. (see above).
 */
template<class Real>
float computeMutualInformationKraskov2(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<Real>& cache) {
    //const int base = 2;
    cache.points.clear();
    cache.pointsCopy.clear();
    cache.kdTree2d.clear();
    cache.kthNeighborDistances.clear();
    cache.nearestNeighborDistances.clear();
    cache.kthNeighborDistancesRef.clear();
    cache.kthNeighborDistancesQuery.clear();
    cache.nearestNeighbors.clear();

#ifdef KRASKOV_USE_RANDOM_NOISE
    XorshiftRandomGenerator genRef(617406168ul);
    XorshiftRandomGenerator genQuery(864730169ul);
#endif

    cache.points.reserve(es);
    cache.pointsCopy.reserve(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        cache.points.emplace_back(
                referenceValues[e] + genRef.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise,
                queryValues[e] + genQuery.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise);
#else
        cache.points.emplace_back(referenceValues[e], queryValues[e]);
#endif
        cache.pointsCopy.push_back(cache.points.back());
    }
    cache.kdTree2d.buildInplace(cache.pointsCopy);

    cache.kthNeighborDistancesRef.reserve(es);
    cache.kthNeighborDistancesQuery.reserve(es);
    cache.nearestNeighborDistances.reserve(k + 1);
    cache.nearestNeighbors.reserve(k + 1);
    for (int e = 0; e < es; e++) {
        cache.nearestNeighborDistances.clear();
        cache.kdTree2d.findKNearestNeighbors(
                cache.points.at(e), k + 1, cache.nearestNeighbors, cache.nearestNeighborDistances);
        Real distX = std::numeric_limits<Real>::lowest();
        Real distY = std::numeric_limits<Real>::lowest();
        for (size_t i = 0; i < cache.nearestNeighbors.size(); i++) {
            Real ex = std::abs(cache.points.at(e).x - cache.nearestNeighbors.at(i).x);
            Real ey = std::abs(cache.points.at(e).y - cache.nearestNeighbors.at(i).y);
            distX = std::max(distX, ex);
            distY = std::max(distY, ey);
        }
        cache.kthNeighborDistancesRef.emplace_back(distX);
        cache.kthNeighborDistancesQuery.emplace_back(distY);
    }

    auto a = averageDigamma<Real, false>(referenceValues, es, cache.kthNeighborDistancesRef, true, cache);
    auto b = averageDigamma<Real, false>(queryValues, es, cache.kthNeighborDistancesQuery, false, cache);
    auto c = Real(digamma(k)) - Real(1) / Real(k);
    auto d = Real(digamma(es));

    //Real mi = (-a - b + c + d) / Real(std::log(base));
    Real mi = -a - b + c + d;
    return float(mi);
}

template
float computeMutualInformationKraskov<float>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<float>& cache);
template
float computeMutualInformationKraskov<double>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<double>& cache);

template
float computeMutualInformationKraskov2<float>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<float>& cache);
template
float computeMutualInformationKraskov2<double>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<double>& cache);
