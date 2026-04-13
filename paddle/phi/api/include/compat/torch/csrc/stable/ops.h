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

#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/select.h>
#include <ATen/ops/to.h>
#include <ATen/ops/transpose.h>

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace torch::stable {

using DeleterFnPtr = void (*)(void*);
using ScalarType = torch::headeronly::ScalarType;
using IntHeaderOnlyArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;

inline Tensor empty(
    IntHeaderOnlyArrayRef size,
    std::optional<ScalarType> dtype = std::nullopt,
    std::optional<at::Layout> layout = std::nullopt,
    std::optional<Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) {
  std::optional<at::ScalarType> at_dtype = std::nullopt;
  if (dtype.has_value()) {
    at_dtype = static_cast<at::ScalarType>(dtype.value());
  }
  return Tensor(
      at::empty(size, at_dtype, layout, device, pin_memory, memory_format));
}

inline Tensor full(IntHeaderOnlyArrayRef size,
                   const at::Scalar& fill_value,
                   std::optional<ScalarType> dtype = std::nullopt,
                   std::optional<at::Layout> layout = std::nullopt,
                   std::optional<Device> device = std::nullopt,
                   std::optional<bool> pin_memory = std::nullopt) {
  std::optional<at::ScalarType> at_dtype = std::nullopt;
  if (dtype.has_value()) {
    at_dtype = static_cast<at::ScalarType>(dtype.value());
  }
  return Tensor(
      at::full(size, fill_value, at_dtype, layout, device, pin_memory));
}

inline Tensor narrow(const Tensor& self,
                     int64_t dim,
                     int64_t start,
                     int64_t length) {
  return Tensor(at::narrow(self._PD_GetInner(), dim, start, length));
}

inline Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  return Tensor(at::select(self._PD_GetInner(), dim, index));
}

inline Tensor to(const Tensor& self, const Device& device) {
  return Tensor(self._PD_GetInner().to(device,
                                       self._PD_GetInner().scalar_type(),
                                       /*non_blocking=*/false,
                                       /*copy=*/false,
                                       std::nullopt));
}

inline Tensor contiguous(const Tensor& self) {
  return Tensor(self._PD_GetInner().contiguous());
}

inline Tensor zero_(Tensor& self) {  // NOLINT(runtime/references)
  self._PD_GetInner().zero_();
  return self;
}

inline Tensor copy_(Tensor& self,  // NOLINT(runtime/references)
                    const Tensor& src,
                    std::optional<bool> non_blocking = std::nullopt) {
  self._PD_GetInner().copy_(src._PD_GetInner(), non_blocking.value_or(false));
  return self;
}

inline Tensor from_blob(void* data,
                        IntHeaderOnlyArrayRef sizes,
                        IntHeaderOnlyArrayRef strides,
                        const Device& device,
                        ScalarType dtype,
                        DeleterFnPtr deleter = nullptr) {
  auto options = at::TensorOptions()
                     .dtype(static_cast<at::ScalarType>(dtype))
                     .device(device);
  if (deleter != nullptr) {
    std::function<void(void*)> deleter_fn = deleter;
    return Tensor(at::from_blob(data, sizes, strides, deleter_fn, options));
  }
  return Tensor(at::from_blob(data, sizes, strides, options));
}

inline Tensor from_blob(void* data,
                        IntHeaderOnlyArrayRef sizes,
                        const Device& device,
                        ScalarType dtype,
                        DeleterFnPtr deleter = nullptr) {
  auto options = at::TensorOptions()
                     .dtype(static_cast<at::ScalarType>(dtype))
                     .device(device);
  if (deleter != nullptr) {
    std::function<void(void*)> deleter_fn = deleter;
    return Tensor(at::from_blob(data, sizes, deleter_fn, options));
  }
  return Tensor(at::from_blob(data, sizes, options));
}

inline Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0) {
  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    at_tensors.push_back(tensor._PD_GetInner());
  }
  return Tensor(at::cat(at_tensors, dim));
}

inline Tensor permute(const Tensor& self, IntHeaderOnlyArrayRef dims) {
  return Tensor(at::permute(self._PD_GetInner(), dims));
}

inline Tensor rot90(const Tensor& self,
                    int64_t k,
                    IntHeaderOnlyArrayRef dims = IntHeaderOnlyArrayRef({0,
                                                                        1})) {
  STD_TORCH_CHECK(dims.size() == 2,
                  "rot90 expects dims to contain exactly 2 dimensions, got ",
                  dims.size());

  const int64_t ndim = self.dim();
  int64_t dim0 = dims[0];
  int64_t dim1 = dims[1];
  if (dim0 < 0) dim0 += ndim;
  if (dim1 < 0) dim1 += ndim;

  STD_TORCH_CHECK(
      dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim && dim0 != dim1,
      "rot90 received invalid dims: (",
      dims[0],
      ", ",
      dims[1],
      ") for tensor dim ",
      ndim);

  int64_t normalized_k = k % 4;
  if (normalized_k < 0) {
    normalized_k += 4;
  }
  if (normalized_k == 0) {
    return self;
  }

  at::Tensor result = self._PD_GetInner();
  if (normalized_k == 2) {
    result = at::Tensor(paddle::experimental::flip(result._PD_GetInner(),
                                                   {static_cast<int>(dim0)}));
    result = at::Tensor(paddle::experimental::flip(result._PD_GetInner(),
                                                   {static_cast<int>(dim1)}));
    return Tensor(std::move(result));
  }

  result = at::transpose(result, dim0, dim1);
  const int flip_dim =
      normalized_k == 1 ? static_cast<int>(dim0) : static_cast<int>(dim1);
  result =
      at::Tensor(paddle::experimental::flip(result._PD_GetInner(), {flip_dim}));
  return Tensor(std::move(result));
}

}  // namespace torch::stable
