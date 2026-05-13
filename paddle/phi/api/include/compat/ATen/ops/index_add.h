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
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

#include "paddle/phi/api/include/api.h"

namespace at {

inline paddle::Tensor _index_add_apply_alpha(const at::Tensor& source,
                                             const at::Scalar& alpha) {
  if (alpha.to<double>() == 1.0) {
    return source._PD_GetInner();
  }
  return paddle::experimental::scale(source._PD_GetInner(),
                                     phi::Scalar(alpha.to<double>()),
                                     /*bias=*/0.0f,
                                     /*bias_after_scale=*/true);
}

// index_add: out-of-place
inline at::Tensor index_add(const at::Tensor& self,
                            int64_t dim,
                            const at::Tensor& index,
                            const at::Tensor& source,
                            const at::Scalar& alpha = 1) {
  auto add_value = _index_add_apply_alpha(source, alpha);
  return paddle::experimental::index_add(self._PD_GetInner(),
                                         index._PD_GetInner(),
                                         add_value,
                                         static_cast<int>(dim));
}

}  // namespace at

namespace at {

inline at::Tensor& Tensor::index_add_(int64_t dim,
                                      const at::Tensor& index,
                                      const at::Tensor& source,
                                      const at::Scalar& alpha) const {
  auto add_value = _index_add_apply_alpha(source, alpha);
  at::Tensor& self_ref = const_cast<at::Tensor&>(*this);
  paddle::experimental::index_add_(self_ref._PD_GetInner(),
                                   index._PD_GetInner(),
                                   add_value,
                                   static_cast<int>(dim));
  return self_ref;
}

inline at::Tensor Tensor::index_add(int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& source,
                                    const at::Scalar& alpha) const {
  return at::index_add(*this, dim, index, source, alpha);
}

}  // namespace at
