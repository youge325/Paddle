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

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch::stable {

using ScalarType = torch::headeronly::ScalarType;
using IntHeaderOnlyArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;

class Tensor {
 public:
  Tensor() : tensor_(std::make_shared<at::Tensor>()) {}

  explicit Tensor(AtenTensorHandle handle) {
    if (handle != nullptr) {
      tensor_.reset(reinterpret_cast<at::Tensor*>(handle),
                    [](at::Tensor* ptr) { delete ptr; });
    } else {
      tensor_ = std::make_shared<at::Tensor>();
    }
  }

  explicit Tensor(const at::Tensor& tensor)
      : tensor_(std::make_shared<at::Tensor>(tensor)) {}

  explicit Tensor(at::Tensor&& tensor)
      : tensor_(std::make_shared<at::Tensor>(std::move(tensor))) {}

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor() = default;

  AtenTensorHandle get() const {
    return reinterpret_cast<AtenTensorHandle>(tensor_.get());
  }

  void* data_ptr() const { return tensor_->data_ptr(); }

  void* mutable_data_ptr() const { return tensor_->mutable_data_ptr(); }

  const void* const_data_ptr() const { return tensor_->const_data_ptr(); }

  template <typename T>
  T* mutable_data_ptr() const {
    return tensor_->template mutable_data_ptr<T>();
  }

  template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
  const T* const_data_ptr() const {
    return tensor_->template const_data_ptr<T>();
  }

  IntHeaderOnlyArrayRef sizes() const {
    return IntHeaderOnlyArrayRef(tensor_->sizes());
  }

  IntHeaderOnlyArrayRef strides() const {
    return IntHeaderOnlyArrayRef(tensor_->strides());
  }

  int64_t size(int64_t dim) const { return tensor_->size(dim); }

  int64_t stride(int64_t dim) const { return tensor_->stride(dim); }

  int64_t dim() const { return tensor_->dim(); }

  int64_t numel() const { return tensor_->numel(); }

  size_t element_size() const { return tensor_->element_size(); }

  bool is_contiguous() const { return tensor_->is_contiguous(); }

  ScalarType scalar_type() const {
    return static_cast<ScalarType>(tensor_->scalar_type());
  }

  Device device() const { return tensor_->device(); }

  bool is_cpu() const { return tensor_->is_cpu(); }

  bool is_cuda() const { return tensor_->is_cuda(); }

  bool defined() const { return tensor_->defined(); }

  operator at::Tensor() const { return *tensor_; }  // NOLINT

  const at::Tensor& _PD_GetInner() const { return *tensor_; }
  at::Tensor& _PD_GetInner() { return *tensor_; }

 private:
  std::shared_ptr<at::Tensor> tensor_;
};

}  // namespace torch::stable

namespace torch {

template <>
inline stable::Tensor generic_to(const IValue& ivalue,
                                 _fake_type<stable::Tensor>) {
  return stable::Tensor(ivalue.toTensor());
}

}  // namespace torch
