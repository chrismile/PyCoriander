# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Sample code demonstrating how to use PyCoriander as a Python module installed with setuptools.

import time
import torch
import numpy as np
import pycoriander

if __name__ == '__main__':
    cuda_device_idx = 0
    cuda_device = torch.device(f'cuda:{cuda_device_idx}' if torch.cuda.is_available() else 'cpu')

    M = 250 * 352 * 20
    N = 100
    np.random.seed(1401)
    reference_array = np.random.uniform(0.0, 1.0, N).astype('f')
    query_array = np.random.uniform(0.0, 1.0, (M, N)).astype('f')
    reference_tensor_cpu = torch.from_numpy(reference_array)
    query_tensor_cpu = torch.from_numpy(query_array)
    reference_tensor_cuda = reference_tensor_cpu.to(cuda_device)
    query_tensor_cuda = query_tensor_cpu.to(cuda_device)


    start_time = time.time()
    output_tensor_cpu = pycoriander.pearson_correlation(reference_tensor_cpu, query_tensor_cpu)
    print(f'Time OpenMP (Pearson): {time.time() - start_time}')

    # First run without measuring time to make sure the kernels are compiled.
    output_tensor_cuda = pycoriander.pearson_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    start_time = time.time()
    output_tensor_cuda = pycoriander.pearson_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    print(f'Time CUDA (Pearson): {time.time() - start_time}')


    start_time = time.time()
    output_tensor_cpu = pycoriander.spearman_rank_correlation(reference_tensor_cpu, query_tensor_cpu)
    print(f'Time OpenMP (Spearman): {time.time() - start_time}')

    # First run without measuring time to make sure the kernels are compiled.
    output_tensor_cuda = pycoriander.spearman_rank_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    start_time = time.time()
    output_tensor_cuda = pycoriander.spearman_rank_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    print(f'Time CUDA (Spearman): {time.time() - start_time}')


    start_time = time.time()
    output_tensor_cpu = pycoriander.kendall_rank_correlation(reference_tensor_cpu, query_tensor_cpu)
    print(f'Time OpenMP (Kendall): {time.time() - start_time}')

    # First run without measuring time to make sure the kernels are compiled.
    output_tensor_cuda = pycoriander.kendall_rank_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    start_time = time.time()
    output_tensor_cuda = pycoriander.kendall_rank_correlation(reference_tensor_cuda, query_tensor_cuda)
    torch.cuda.synchronize(cuda_device)
    print(f'Time CUDA (Kendall): {time.time() - start_time}')


    start_time = time.time()
    output_tensor_cpu = pycoriander.mutual_information_binned(
            reference_tensor_cpu, query_tensor_cpu, 80, 0.0, 1.0, 0.0, 1.0)
    print(f'Time OpenMP (MI binned): {time.time() - start_time}')

    # First run without measuring time to make sure the kernels are compiled.
    output_tensor_cuda = pycoriander.mutual_information_binned(
            reference_tensor_cuda, query_tensor_cuda, 80, 0.0, 1.0, 0.0, 1.0)
    torch.cuda.synchronize(cuda_device)
    start_time = time.time()
    output_tensor_cuda = pycoriander.mutual_information_binned(
            reference_tensor_cuda, query_tensor_cuda, 80, 0.0, 1.0, 0.0, 1.0)
    torch.cuda.synchronize(cuda_device)
    print(f'Time CUDA (MI binned): {time.time() - start_time}')


    start_time = time.time()
    output_tensor_cpu = pycoriander.mutual_information_kraskov(reference_tensor_cpu, query_tensor_cpu, 3)
    print(f'Time OpenMP (MI Kraskov): {time.time() - start_time}')

    # First run without measuring time to make sure the kernels are compiled.
    output_tensor_cuda = pycoriander.mutual_information_kraskov(reference_tensor_cuda, query_tensor_cuda, 3)
    torch.cuda.synchronize(cuda_device)
    start_time = time.time()
    output_tensor_cuda = pycoriander.mutual_information_kraskov(reference_tensor_cuda, query_tensor_cuda, 3)
    torch.cuda.synchronize(cuda_device)
    print(f'Time CUDA (MI Kraskov): {time.time() - start_time}')
