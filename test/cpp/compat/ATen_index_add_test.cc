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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/index_add.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

namespace {

at::Tensor MakeIndex(const std::vector<int64_t>& vals,
                     at::ScalarType dt = at::kLong) {
  at::Tensor idx = at::empty({static_cast<int64_t>(vals.size())}, dt);
  if (dt == at::kLong) {
    int64_t* p = idx.data_ptr<int64_t>();
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
  } else if (dt == at::kInt) {
    int32_t* p = idx.data_ptr<int32_t>();
    for (size_t i = 0; i < vals.size(); ++i)
      p[i] = static_cast<int32_t>(vals[i]);
  }
  return idx;
}

}  // namespace

TEST(TensorIndexAddTest, FreeFunctionDefaultAlpha) {
  at::Tensor self = at::zeros({5}, at::kFloat);
  at::Tensor index = MakeIndex({0, 2, 4});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor out = at::index_add(self, 0, index, source);
  ASSERT_EQ(out.numel(), 5);

  float* d = out.data_ptr<float>();
  ASSERT_FLOAT_EQ(d[0], 2.0f);
  ASSERT_FLOAT_EQ(d[1], 0.0f);
  ASSERT_FLOAT_EQ(d[2], 2.0f);
  ASSERT_FLOAT_EQ(d[3], 0.0f);
  ASSERT_FLOAT_EQ(d[4], 2.0f);
}

TEST(TensorIndexAddTest, MethodOutOfPlaceDoesNotMutateSelf) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  float* self_ptr = self.data_ptr<float>();
  at::Tensor index = MakeIndex({0, 1});
  at::Tensor source = at::full({2}, 3.0f, at::kFloat);

  at::Tensor out = self.index_add(0, index, source);
  ASSERT_FLOAT_EQ(self_ptr[0], 0.0f);
  ASSERT_FLOAT_EQ(out.data_ptr<float>()[0], 3.0f);
  ASSERT_FLOAT_EQ(out.data_ptr<float>()[1], 3.0f);
}

TEST(TensorIndexAddTest, MethodInplaceMutatesSelf) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  float* original_ptr = self.data_ptr<float>();
  at::Tensor index = MakeIndex({0, 2});
  at::Tensor source = at::full({2}, 5.0f, at::kFloat);

  at::Tensor& ref = self.index_add_(0, index, source);
  ASSERT_EQ(ref.data_ptr<float>(), original_ptr);
  ASSERT_FLOAT_EQ(self.data_ptr<float>()[0], 5.0f);
  ASSERT_FLOAT_EQ(self.data_ptr<float>()[2], 5.0f);
}

TEST(TensorIndexAddTest, AlphaTwoScalesSource) {
  at::Tensor self = at::zeros({3}, at::kFloat);
  at::Tensor index = MakeIndex({0, 1, 2});
  at::Tensor source = at::full({3}, 1.5f, at::kFloat);

  at::Tensor out = at::index_add(self, 0, index, source, at::Scalar(2.0));
  float* d = out.data_ptr<float>();
  ASSERT_FLOAT_EQ(d[0], 3.0f);
  ASSERT_FLOAT_EQ(d[1], 3.0f);
  ASSERT_FLOAT_EQ(d[2], 3.0f);
}

TEST(TensorIndexAddTest, AlphaNegativeSubtracts) {
  at::Tensor self = at::full({3}, 10.0f, at::kFloat);
  at::Tensor index = MakeIndex({0, 1, 2});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor out = at::index_add(self, 0, index, source, at::Scalar(-1.0));
  float* d = out.data_ptr<float>();
  ASSERT_FLOAT_EQ(d[0], 8.0f);
  ASSERT_FLOAT_EQ(d[1], 8.0f);
  ASSERT_FLOAT_EQ(d[2], 8.0f);
}

TEST(TensorIndexAddTest, NegativeDimWrapsCorrectly) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = MakeIndex({0, 2});
  at::Tensor source = at::full({2, 2}, 7.0f, at::kFloat);

  at::Tensor out_pos = at::index_add(self, 1, index, source);
  at::Tensor out_neg = at::index_add(self, -1, index, source);
  ASSERT_TRUE(out_pos.equal(out_neg));
}

TEST(TensorIndexAddTest, IndexInt32Accepted) {
  at::Tensor self = at::zeros({4}, at::kFloat);
  at::Tensor index = MakeIndex({0, 3}, at::kInt);
  at::Tensor source = at::full({2}, 1.0f, at::kFloat);

  at::Tensor out = at::index_add(self, 0, index, source);
  ASSERT_FLOAT_EQ(out.data_ptr<float>()[0], 1.0f);
  ASSERT_FLOAT_EQ(out.data_ptr<float>()[3], 1.0f);
}

TEST(TensorIndexAddTest, IntegerSelfFloatAlphaDoesNotThrow) {
  at::Tensor self = at::zeros({3}, at::kLong);
  at::Tensor index = MakeIndex({0, 1});
  at::Tensor source = at::ones({2}, at::kLong);

  // libtorch CPU does not enforce alpha-must-be-integral on integral self;
  // we mirror that relaxed behavior in the Paddle compat wrapper.
  at::Tensor out = at::index_add(self, 0, index, source, at::Scalar(1.5));
  ASSERT_EQ(out.numel(), 3);
}

TEST(TensorIndexAddTest, RepeatedIndexAccumulates) {
  at::Tensor self = at::zeros({3}, at::kFloat);
  at::Tensor index = MakeIndex({0, 0, 0});
  at::Tensor source = at::full({3}, 2.0f, at::kFloat);

  at::Tensor out = at::index_add(self, 0, index, source);
  ASSERT_FLOAT_EQ(out.data_ptr<float>()[0], 6.0f);
}
