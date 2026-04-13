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

#include <ATen/core/ivalue.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/transpose.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/TensorAccessor.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace torch::stable {

using StableIValue = torch::IValue;

namespace detail {

template <typename T>
inline StableIValue from(T&& value) {
  return StableIValue(std::forward<T>(value));
}

inline StableIValue from(const Tensor& value) {
  return StableIValue(value._PD_GetInner());
}

inline StableIValue from(const std::vector<Tensor>& value) {
  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(value.size());
  for (const auto& tensor : value) {
    at_tensors.push_back(tensor._PD_GetInner());
  }
  return StableIValue(std::move(at_tensors));
}

template <typename T>
inline T to(const StableIValue& value) {
  return value.to<T>();
}

template <>
inline Tensor to<Tensor>(const StableIValue& value) {
  return Tensor(value.to<at::Tensor>());
}

template <>
inline std::vector<Tensor> to<std::vector<Tensor>>(const StableIValue& value) {
  std::vector<at::Tensor> at_tensors = value.to<std::vector<at::Tensor>>();
  std::vector<Tensor> tensors;
  tensors.reserve(at_tensors.size());
  for (auto& tensor : at_tensors) {
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}

}  // namespace detail

}  // namespace torch::stable

using StableIValue = torch::stable::StableIValue;

inline AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* /*overloadName*/,
    torch::stable::StableIValue* stack,
    uint64_t /*extension_build_version*/) {
  if (opName == nullptr || stack == nullptr) {
    return TORCH_FAILURE;
  }

  try {
    const std::string op_name(opName);
    if (op_name == "aten::permute") {
      auto self = torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
      auto dims = torch::stable::detail::to<std::vector<int64_t>>(stack[1]);
      stack[0] = torch::stable::detail::from(at::permute(
          self._PD_GetInner(), torch::headeronly::IntHeaderOnlyArrayRef(dims)));
      return TORCH_SUCCESS;
    }

    if (op_name == "aten::cat") {
      auto tensors =
          torch::stable::detail::to<std::vector<torch::stable::Tensor>>(
              stack[0]);
      const auto dim = torch::stable::detail::to<int64_t>(stack[1]);
      std::vector<at::Tensor> at_tensors;
      at_tensors.reserve(tensors.size());
      for (const auto& tensor : tensors) {
        at_tensors.push_back(tensor._PD_GetInner());
      }
      stack[0] = torch::stable::detail::from(at::cat(at_tensors, dim));
      return TORCH_SUCCESS;
    }

    if (op_name == "aten::rot90") {
      auto self = torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
      int64_t k = torch::stable::detail::to<int64_t>(stack[1]);
      auto dims = torch::stable::detail::to<std::vector<int64_t>>(stack[2]);

      TORCH_CHECK(dims.size() == 2,
                  "aten::rot90 expects exactly 2 dims, got ",
                  dims.size());

      int64_t dim0 = dims[0];
      int64_t dim1 = dims[1];
      const int64_t ndim = self.dim();
      if (dim0 < 0) dim0 += ndim;
      if (dim1 < 0) dim1 += ndim;
      TORCH_CHECK(
          dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim && dim0 != dim1,
          "aten::rot90 got invalid dims (",
          dims[0],
          ", ",
          dims[1],
          ") for tensor dim ",
          ndim);

      int64_t normalized_k = k % 4;
      if (normalized_k < 0) {
        normalized_k += 4;
      }

      at::Tensor result = self._PD_GetInner();
      if (normalized_k == 1 || normalized_k == 3) {
        result = at::transpose(result, dim0, dim1);
        const int flip_dim =
            normalized_k == 1 ? static_cast<int>(dim0) : static_cast<int>(dim1);
        result = at::Tensor(
            paddle::experimental::flip(result._PD_GetInner(), {flip_dim}));
      } else if (normalized_k == 2) {
        result = at::Tensor(paddle::experimental::flip(
            result._PD_GetInner(), {static_cast<int>(dim0)}));
        result = at::Tensor(paddle::experimental::flip(
            result._PD_GetInner(), {static_cast<int>(dim1)}));
      }

      stack[0] = torch::stable::detail::from(std::move(result));
      return TORCH_SUCCESS;
    }
  } catch (...) {
    return TORCH_FAILURE;
  }

  return TORCH_FAILURE;
}

#define TORCH_BOX(func) (func)

#define STABLE_TORCH_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define STABLE_TORCH_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)
#define STABLE_TORCH_LIBRARY_FRAGMENT(ns, m) TORCH_LIBRARY_FRAGMENT(ns, m)
