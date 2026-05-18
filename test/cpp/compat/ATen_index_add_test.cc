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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

TEST(IndexAddTest, BasicIndexAdd) {
  at::Tensor t = at::zeros({5, 3}, at::kFloat);

  at::Tensor idx = at::empty({3}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 2;
  idx_data[2] = 4;

  at::Tensor source = at::full({3, 3}, 1.0f, at::kFloat);

  at::Tensor result = t.index_add(0, idx, source);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({5, 3}));

  float* data = result.data_ptr<float>();
  // Row 0: all 1s
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 1.0f);
  // Row 1: all 0s
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 0.0f);
  ASSERT_FLOAT_EQ(data[5], 0.0f);
  // Row 2: all 1s
  ASSERT_FLOAT_EQ(data[6], 1.0f);
  ASSERT_FLOAT_EQ(data[7], 1.0f);
  ASSERT_FLOAT_EQ(data[8], 1.0f);
}

TEST(IndexAddTest, IndexAddWithAlpha) {
  at::Tensor t = at::zeros({5}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 1;
  idx_data[1] = 3;

  at::Tensor source = at::full({2}, 2.0f, at::kFloat);

  at::Tensor result = t.index_add(0, idx, source, 3);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 6.0f);  // 2 * 3
  ASSERT_FLOAT_EQ(data[2], 0.0f);
  ASSERT_FLOAT_EQ(data[3], 6.0f);  // 2 * 3
  ASSERT_FLOAT_EQ(data[4], 0.0f);
}

TEST(IndexAddTest, IndexAddInplace) {
  at::Tensor t = at::zeros({5}, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 4;

  at::Tensor source = at::full({2}, 5.0f, at::kFloat);

  t.index_add_(0, idx, source);

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 5.0f);
  ASSERT_FLOAT_EQ(data[1], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 5.0f);
}

TEST(IndexAddTest, IndexAddDim1) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 2;

  at::Tensor source = at::full({3, 2}, 1.0f, at::kFloat);

  at::Tensor result = t.index_add(1, idx, source);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));

  float* data = result.data_ptr<float>();
  // Col 0: all 1s
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[4], 1.0f);
  ASSERT_FLOAT_EQ(data[8], 1.0f);
  // Col 1: all 0s
  ASSERT_FLOAT_EQ(data[1], 0.0f);
  ASSERT_FLOAT_EQ(data[5], 0.0f);
  ASSERT_FLOAT_EQ(data[9], 0.0f);
  // Col 2: all 1s
  ASSERT_FLOAT_EQ(data[2], 1.0f);
  ASSERT_FLOAT_EQ(data[6], 1.0f);
  ASSERT_FLOAT_EQ(data[10], 1.0f);
}

TEST(IndexAddTest, IndexAddInt64Dtype) {
  at::Tensor t = at::zeros({5}, at::kLong);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 1;
  idx_data[1] = 3;

  at::Tensor source = at::full({2}, 10L, at::kLong);

  at::Tensor result = t.index_add(0, idx, source);

  int64_t* data = result.data_ptr<int64_t>();
  ASSERT_EQ(data[0], 0L);
  ASSERT_EQ(data[1], 10L);
  ASSERT_EQ(data[2], 0L);
  ASSERT_EQ(data[3], 10L);
  ASSERT_EQ(data[4], 0L);
}

TEST(IndexAddTest, IndexAddNegativeIndex) {
  at::Tensor t = at::zeros({5}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = -1;
  idx_data[1] = -3;

  at::Tensor source = at::full({2}, 1.0f, at::kFloat);

  at::Tensor result = t.index_add(0, idx, source);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 0.0f);
  ASSERT_FLOAT_EQ(data[2], 1.0f);  // index -3 -> 2
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 1.0f);  // index -1 -> 4
}

TEST(IndexAddTest, IndexAddAccumulate) {
  at::Tensor t = at::ones({5}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 1;
  idx_data[1] = 1;

  at::Tensor source = at::full({2}, 2.0f, at::kFloat);

  at::Tensor result = t.index_add(0, idx, source);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 5.0f);  // 1 + 2 + 2 (accumulated twice at index 1)
  ASSERT_FLOAT_EQ(data[2], 1.0f);
  ASSERT_FLOAT_EQ(data[3], 1.0f);
  ASSERT_FLOAT_EQ(data[4], 1.0f);
}
