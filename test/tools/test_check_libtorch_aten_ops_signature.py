# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import sys
import tempfile
import textwrap
import unittest

TOOLS_PATH = pathlib.Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

from check_libtorch_aten_ops_signature import (
    SignatureIssue,
    collect_signature_issues,
    compare_with_baseline,
    load_baseline,
    read_torch_version,
    write_baseline,
)


class AtenOpsSignatureCheckTest(unittest.TestCase):
    def _write(self, root, relative_path, content):
        path = pathlib.Path(root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
        return path

    def _write_common_tree(self, paddle_root, libtorch_include):
        self._write(
            paddle_root,
            "paddle/phi/api/include/compat/ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              Tensor foo(
                  std::optional<ScalarType> dtype = std::nullopt) const;
             protected:
              at::Tensor hidden() const;
            };
            }  // namespace at
            """,
        )
        self._write(
            libtorch_include,
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class TORCH_API Tensor: public TensorBase {
             public:
              at::Tensor foo(
                  ::std::optional<at::ScalarType> dtype=::std::nullopt) const;
              at::Tensor foo(int64_t dim=0) const;
            };
            }  // namespace at
            """,
        )

    def test_matching_free_function_and_tensor_member_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            paddle_root = tmp_path / "paddle-root"
            libtorch_include = tmp_path / "torch" / "include"
            self._write_common_tree(paddle_root, libtorch_include)
            self._write(
                libtorch_include,
                "ATen/ops/foo.h",
                """
                namespace at {
                inline at::Tensor foo(
                    const at::Tensor & self,
                    ::std::optional<at::ScalarType> dtype=::std::nullopt) {
                  return self;
                }
                inline at::Tensor foo(const at::Tensor& self, int64_t dim=0) {
                  return self;
                }
                }  // namespace at
                """,
            )
            self._write(
                paddle_root,
                "paddle/phi/api/include/compat/ATen/ops/foo.h",
                """
                namespace at {
                inline Tensor foo(
                    const Tensor& self,
                    std::optional<ScalarType> dtype = std::nullopt) {
                  return self;
                }
                inline int helper(int value) {
                  return value;
                }
                }  // namespace at

                namespace at {
                inline at::Tensor Tensor::foo(
                    std::optional<ScalarType> dtype) const {
                  return at::foo(*this, dtype);
                }
                }  // namespace at
                """,
            )

            self.assertEqual(
                collect_signature_issues(paddle_root, libtorch_include), []
            )

    def test_free_function_default_parameter_difference_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            paddle_root = tmp_path / "paddle-root"
            libtorch_include = tmp_path / "torch" / "include"
            self._write_common_tree(paddle_root, libtorch_include)
            self._write(
                libtorch_include,
                "ATen/ops/bar.h",
                """
                namespace at {
                inline at::Tensor bar(
                    const at::Tensor & self, bool keepdim=false) {
                  return self;
                }
                }  // namespace at
                """,
            )
            self._write(
                paddle_root,
                "paddle/phi/api/include/compat/ATen/ops/bar.h",
                """
                namespace at {
                inline at::Tensor bar(
                    const at::Tensor& self, bool keepdim=true) {
                  return self;
                }
                }  // namespace at
                """,
            )

            issues = collect_signature_issues(paddle_root, libtorch_include)

            self.assertEqual(len(issues), 1)
            self.assertEqual(
                issues[0].title,
                "bar.h: ATen free function signatures differ",
            )
            self.assertEqual(issues[0].missing, ())
            self.assertEqual(len(issues[0].extra), 1)
            self.assertIn("keepdim=true", issues[0].extra[0])

    def test_tensor_member_definition_and_declaration_difference_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            paddle_root = tmp_path / "paddle-root"
            libtorch_include = tmp_path / "torch" / "include"
            self._write(
                libtorch_include,
                "ATen/core/TensorBody.h",
                """
                namespace at {
                class Tensor {
                 public:
                  at::Tensor baz(int64_t dim=0) const;
                };
                }  // namespace at
                """,
            )
            self._write(
                paddle_root,
                "paddle/phi/api/include/compat/ATen/core/TensorBody.h",
                """
                namespace at {
                class Tensor {
                 public:
                  const at::Tensor& baz(int64_t dim) const;
                };
                }  // namespace at
                """,
            )
            self._write(
                libtorch_include,
                "ATen/ops/baz.h",
                """
                namespace at {
                inline at::Tensor baz(const at::Tensor & self, int64_t dim=0) {
                  return self;
                }
                }  // namespace at
                """,
            )
            self._write(
                paddle_root,
                "paddle/phi/api/include/compat/ATen/ops/baz.h",
                """
                namespace at {
                inline at::Tensor baz(const at::Tensor& self, int64_t dim=0) {
                  return self;
                }
                }  // namespace at

                namespace at {
                inline const at::Tensor& Tensor::baz(int64_t dim) const {
                  return *this;
                }
                }  // namespace at
                """,
            )

            issue_titles = [
                issue.title
                for issue in collect_signature_issues(
                    paddle_root, libtorch_include
                )
            ]

            self.assertIn(
                "baz.h: Tensor::baz definition signatures differ",
                issue_titles,
            )
            self.assertIn(
                "baz.h: Tensor::baz declaration signatures differ in "
                "compat/ATen/core/TensorBody.h",
                issue_titles,
            )

    def test_reads_multiline_torch_version_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            include = pathlib.Path(tmp)
            self._write(
                include,
                "torch/csrc/api/include/torch/version.h",
                '''
                #define TORCH_VERSION_MAJOR 2
                #define TORCH_VERSION_MINOR 9
                #define TORCH_VERSION_PATCH 1
                #define TORCH_VERSION \\
                  "2.9.1"
                ''',
            )

            self.assertEqual(read_torch_version(include), "2.9.1")

    def test_baseline_compare_detects_new_and_resolved_differences(self):
        known = SignatureIssue("known", ("missing",), ())
        new = SignatureIssue("new", (), ("extra",))

        self.assertEqual(compare_with_baseline([known], [known]), [])
        self.assertIn(
            "New or changed Paddle signatures not found in libtorch",
            "\n".join(compare_with_baseline([known, new], [known])),
        )
        self.assertIn(
            "Baseline differences no longer present",
            "\n".join(compare_with_baseline([known], [known, new])),
        )

    def test_baseline_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "baseline.json"
            issues = [SignatureIssue("known", ("missing",), ("extra",))]

            write_baseline(path, issues, "2.9.1")

            self.assertEqual(load_baseline(path), issues)


if __name__ == "__main__":
    unittest.main()
