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

#include <cstdint>
#include <string>

namespace c10 {

// Compat no-op command line flag APIs. This keeps call sites source-compatible
// without introducing a gflags dependency into the header-only compat layer.
inline void SetUsageMessage(const std::string& /*str*/) {}

inline const char* UsageMessage() { return ""; }

inline bool ParseCommandLineFlags(int* /*pargc*/, char*** /*pargv*/) {
  return true;
}

inline bool CommandLineFlagsHasBeenParsed() { return true; }

}  // namespace c10

#define C10_DEFINE_typed_var(type, name, default_value, help_str) \
  type FLAGS_##name = (default_value)

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_DEFINE_typed_var(int, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_DEFINE_typed_var(double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_DEFINE_typed_var(bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_DEFINE_typed_var(std::string, name, default_value, help_str)

#define C10_DECLARE_typed_var(type, name) extern type FLAGS_##name
#define C10_DECLARE_int(name) C10_DECLARE_typed_var(int, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) C10_DECLARE_typed_var(int64_t, name)
#define C10_DECLARE_double(name) C10_DECLARE_typed_var(double, name)
#define C10_DECLARE_bool(name) C10_DECLARE_typed_var(bool, name)
#define C10_DECLARE_string(name) C10_DECLARE_typed_var(std::string, name)

#define TORCH_DECLARE_typed_var(type, name) extern type FLAGS_##name
#define TORCH_DECLARE_int(name) TORCH_DECLARE_typed_var(int, name)
#define TORCH_DECLARE_int32(name) TORCH_DECLARE_int(name)
#define TORCH_DECLARE_int64(name) TORCH_DECLARE_typed_var(int64_t, name)
#define TORCH_DECLARE_double(name) TORCH_DECLARE_typed_var(double, name)
#define TORCH_DECLARE_bool(name) TORCH_DECLARE_typed_var(bool, name)
#define TORCH_DECLARE_string(name) TORCH_DECLARE_typed_var(std::string, name)
