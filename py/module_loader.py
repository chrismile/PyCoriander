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

# Helper functions for loading C++ TorchLib modules.

import os
import torch


modules_path = 'modules/'
loaded_modules = set()


def get_modules_path():
    return modules_path


def set_modules_path(new_modules_path):
    global modules_path
    modules_path = new_modules_path


def get_module_path(module_name):
    if os.name == 'nt':
        return modules_path + module_name + '.dll'
    else:
        return modules_path + 'lib' + module_name + '.so'


def module_exists(module_name):
    return os.path.isfile(get_module_path(module_name))


def is_module_loaded(module_name):
    return module_name in loaded_modules


def load_module(module_name):
    if not module_name in loaded_modules:
        torch.ops.load_library(get_module_path(module_name))
        loaded_modules.add(module_name)
