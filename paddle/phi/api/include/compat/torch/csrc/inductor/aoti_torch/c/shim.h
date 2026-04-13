// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <c10/core/Device.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAStream.h>
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

using AOTITorchError = int32_t;

#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

#define TORCH_SUCCESS AOTI_TORCH_SUCCESS
#define TORCH_FAILURE AOTI_TORCH_FAILURE

#ifndef TORCH_ABI_VERSION
#define TORCH_ABI_VERSION 0ULL
#endif

inline AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index,
                                                         void** ret_stream) {
  if (ret_stream == nullptr) {
    return AOTI_TORCH_FAILURE;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto stream = c10::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(device_index));
#ifdef PADDLE_WITH_HIP
  *ret_stream = reinterpret_cast<void*>(stream.raw_stream());
#else
  *ret_stream = reinterpret_cast<void*>(stream.raw_stream());
#endif
#else
  (void)device_index;
  *ret_stream = nullptr;
#endif
  return AOTI_TORCH_SUCCESS;
}

#ifndef TORCH_ERROR_CODE_CHECK
#define TORCH_ERROR_CODE_CHECK(call)                            \
  do {                                                          \
    if ((call) != TORCH_SUCCESS) {                              \
      throw std::runtime_error(std::string(#call) + " failed"); \
    }                                                           \
  } while (false)
#endif
