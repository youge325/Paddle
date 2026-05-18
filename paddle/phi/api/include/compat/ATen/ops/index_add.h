// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/Scalar.h>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor index_add(const at::Tensor& self,
                            int64_t dim,
                            const at::Tensor& index,
                            const at::Tensor& source,
                            const at::Scalar& alpha = 1) {
  paddle::Tensor pd_source = source._PD_GetInner();
  if (alpha.to<double>() != 1.0) {
    pd_source = paddle::experimental::scale(pd_source, alpha);
  }
  return paddle::experimental::index_add(self._PD_GetInner(),
                                         index._PD_GetInner(),
                                         pd_source,
                                         static_cast<int>(dim));
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::index_add(int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& source,
                                    const at::Scalar& alpha) const {
  return at::index_add(*this, dim, index, source, alpha);
}

inline at::Tensor& Tensor::index_add_(int64_t dim,
                                      const at::Tensor& index,
                                      const at::Tensor& source,
                                      const at::Scalar& alpha) const {
  paddle::Tensor pd_source = source._PD_GetInner();
  if (alpha.to<double>() != 1.0) {
    pd_source = paddle::experimental::scale(pd_source, alpha);
  }
  paddle::experimental::index_add_(const_cast<PaddleTensor&>(tensor_),
                                   index._PD_GetInner(),
                                   pd_source,
                                   static_cast<int>(dim));
  return const_cast<at::Tensor&>(*this);
}

}  // namespace at
