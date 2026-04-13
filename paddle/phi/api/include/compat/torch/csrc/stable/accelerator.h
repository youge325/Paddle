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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <cstdint>

namespace torch::stable::accelerator {

using DeviceIndex = int32_t;
using StreamId = int64_t;

class DeviceGuard {
 public:
  explicit DeviceGuard(DeviceIndex device_index)
      :
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        guard_(static_cast<c10::DeviceIndex>(device_index))
#else
        device_index_(device_index)
#endif
  {
  }

  void set_index(DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    guard_.set_index(static_cast<c10::DeviceIndex>(device_index));
#else
    device_index_ = device_index;
#endif
  }

 private:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c10::cuda::CUDAGuard guard_;
#else
  DeviceIndex device_index_{0};
#endif
};

class Stream {
 public:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  explicit Stream(c10::cuda::CUDAStream stream) : stream_(stream) {}
#else
  explicit Stream(StreamId stream_id) : stream_id_(stream_id) {}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  StreamId id() const { return static_cast<StreamId>(stream_.id()); }
#else
  StreamId id() const { return stream_id_; }
#endif

 private:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c10::cuda::CUDAStream stream_;
#else
  StreamId stream_id_{0};
#endif
};

inline Stream getCurrentStream(DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return Stream(c10::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(device_index)));
#else
  (void)device_index;
  return Stream(0);
#endif
}

inline DeviceIndex getCurrentDeviceIndex() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return static_cast<DeviceIndex>(
      c10::cuda::getCurrentCUDAStream().device_index());
#else
  return 0;
#endif
}

}  // namespace torch::stable::accelerator
