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

"""Check Paddle ATen compat op signatures against libtorch headers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXPECTED_TORCH_VERSION = "2.9.1"
ATEN_OPS_REL = Path("ATen/ops")
TENSOR_BODY_REL = Path("ATen/core/TensorBody.h")
TORCH_VERSION_RELS = (
    Path("torch/csrc/api/include/torch/version.h"),
    Path("torch/headeronly/version.h"),
)
PADDLE_COMPAT_REL = Path("paddle/phi/api/include/compat")
PADDLE_OPS_REL = PADDLE_COMPAT_REL / "ATen/ops"
PADDLE_TENSOR_BODY_REL = PADDLE_COMPAT_REL / "ATen/core/TensorBody.h"
DEFAULT_BASELINE = (
    Path(__file__)
    .resolve()
    .with_name("compat_aten_ops_signature_baseline.json")
)

_KNOWN_ATEN_ALIASES = (
    "Tensor",
    "TensorList",
    "ITensorListRef",
    "IntArrayRef",
    "OptionalIntArrayRef",
    "OptionalTensorRef",
    "TensorOptions",
    "ScalarType",
    "Scalar",
    "Layout",
    "Device",
    "MemoryFormat",
    "Dimname",
    "DimnameList",
    "Stream",
    "Generator",
)


@dataclass(frozen=True)
class SignatureIssue:
    title: str
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "title": self.title,
            "missing": list(self.missing),
            "extra": list(self.extra),
        }

    @staticmethod
    def from_json(data: dict[str, object]) -> SignatureIssue:
        return SignatureIssue(
            title=str(data["title"]),
            missing=tuple(str(item) for item in data.get("missing", [])),
            extra=tuple(str(item) for item in data.get("extra", [])),
        )


def strip_comments(text: str) -> str:
    result = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ('"', "'"):
            quote = char
            result.append(char)
            index += 1
        elif char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
        elif char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (
                text[index] == "*" and text[index + 1] == "/"
            ):
                index += 1
            index += 2
        else:
            result.append(char)
            index += 1
    return "".join(result)


def strip_preprocessor_lines(text: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith("#") else line
        for line in text.splitlines()
    )


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ('"', "'"):
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unmatched '{{' at index {open_index}")


def find_named_blocks(text: str, pattern: str) -> list[str]:
    blocks = []
    for match in re.finditer(pattern, text):
        open_index = text.find("{", match.start(), match.end() + 1)
        if open_index < 0:
            continue
        close_index = find_matching_brace(text, open_index)
        blocks.append(text[open_index + 1 : close_index])
    return blocks


def namespace_blocks(text: str, namespace: str) -> list[str]:
    return find_named_blocks(
        text, rf"\bnamespace\s+{re.escape(namespace)}\s*\{{"
    )


def tensor_class_blocks(text: str) -> list[str]:
    return find_named_blocks(
        text,
        r"\bclass\s+(?:[A-Za-z_]\w*\s+)*Tensor\b[^;{]*\{",
    )


def split_top_level_declarations(block: str) -> list[tuple[str, str]]:
    declarations: list[tuple[str, str]] = []
    start = 0
    index = 0
    paren_depth = 0
    quote: str | None = None
    escaped = False

    while index < len(block):
        char = block[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ('"', "'"):
            quote = char
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif char == ";" and paren_depth == 0:
            segment = block[start:index].strip()
            if segment:
                declarations.append(("declaration", segment))
            start = index + 1
        elif char == "{" and paren_depth == 0:
            segment = block[start:index].strip()
            if segment:
                declarations.append(("definition", segment))
            index = find_matching_brace(block, index)
            start = index + 1
        index += 1

    tail = block[start:].strip()
    if tail:
        declarations.append(("declaration", tail))
    return declarations


def strip_leading_access_specifiers(signature: str) -> str:
    return re.sub(
        r"^(?:\s*(?:public|private|protected)\s*:\s*)+",
        "",
        signature.strip(),
    )


def _looks_like_function_signature(signature: str) -> bool:
    signature = strip_leading_access_specifiers(signature)
    if "(" not in signature or ")" not in signature:
        return False
    first_token = signature.split(None, 1)[0] if signature.split() else ""
    return first_token not in {
        "class",
        "struct",
        "namespace",
        "enum",
        "using",
        "typedef",
        "static_assert",
    }


def _normalize_at_aliases(signature: str) -> str:
    for name in _KNOWN_ATEN_ALIASES:
        signature = re.sub(
            rf"(?<![:\w]){name}\b(?!\s*::)",
            f"at::{name}",
            signature,
        )
    return signature


def normalize_signature(
    signature: str, normalize_at_aliases: bool = True
) -> str:
    signature = strip_leading_access_specifiers(signature)
    signature = re.sub(r"\s+", " ", signature.strip())
    signature = signature.replace("::std::", "std::")
    if normalize_at_aliases:
        signature = _normalize_at_aliases(signature)
    signature = re.sub(r"\s+([&*])", r"\1", signature)
    signature = re.sub(r"([&*])\s+", r"\1 ", signature)
    signature = re.sub(r"\s*::\s*", "::", signature)
    signature = re.sub(r"\s*([(),=<>])\s*", r"\1", signature)
    signature = re.sub(r"=\{\s*\}", "={}", signature)
    signature = re.sub(r"\s+", " ", signature.strip())
    return signature


def strip_parameter_defaults(signature: str) -> str:
    result = []
    paren_depth = 0
    angle_depth = 0
    index = 0
    while index < len(signature):
        char = signature[index]
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(angle_depth - 1, 0)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)

        if char == "=" and paren_depth > 0 and angle_depth == 0:
            index += 1
            nested_parens = 0
            nested_angles = 0
            while index < len(signature):
                next_char = signature[index]
                if next_char == "<":
                    nested_angles += 1
                elif next_char == ">":
                    nested_angles = max(nested_angles - 1, 0)
                elif next_char == "(":
                    nested_parens += 1
                elif next_char == ")":
                    if nested_parens == 0 and nested_angles == 0:
                        break
                    nested_parens = max(nested_parens - 1, 0)
                elif (
                    next_char == ","
                    and nested_parens == 0
                    and nested_angles == 0
                ):
                    break
                index += 1
            continue

        result.append(char)
        index += 1
    return "".join(result)


def class_declaration_to_definition_signature(
    declaration: str, method_name: str
) -> str:
    declaration = strip_parameter_defaults(declaration)
    marker = f"{method_name}("
    index = declaration.rfind(marker)
    if index < 0:
        definition = declaration
    else:
        definition = declaration[:index] + "Tensor::" + declaration[index:]

    if not definition.startswith("template<"):
        definition = re.sub(r"^(?!inline )", "inline ", definition)
    return normalize_signature(definition)


def signature_name(signature: str) -> str | None:
    member_match = re.search(
        r"\bTensor::(?P<name>[A-Za-z_]\w*)\s*\(", signature
    )
    if member_match:
        return member_match.group("name")

    matches = list(re.finditer(r"\b(?P<name>[A-Za-z_]\w*)\s*\(", signature))
    if not matches:
        return None
    return matches[-1].group("name")


def member_definition_name(signature: str) -> str | None:
    match = re.search(r"\bTensor::(?P<name>[A-Za-z_]\w*)\s*\(", signature)
    return match.group("name") if match else None


def class_member_name(signature: str) -> str | None:
    if "Tensor::" in signature:
        return None
    return signature_name(signature)


def _prepared_header_text(header: Path) -> str:
    return strip_preprocessor_lines(strip_comments(header.read_text()))


def extract_free_functions(header: Path) -> set[str]:
    text = _prepared_header_text(header)
    signatures: set[str] = set()
    for at_block in namespace_blocks(text, "at"):
        for kind, segment in split_top_level_declarations(at_block):
            if kind != "definition":
                continue
            if "Tensor::" in segment:
                continue
            if not _looks_like_function_signature(segment):
                continue
            signatures.add(f"at::{normalize_signature(segment)}")

        for symint_block in namespace_blocks(at_block, "symint"):
            for kind, segment in split_top_level_declarations(symint_block):
                if kind != "definition":
                    continue
                if "Tensor::" in segment:
                    continue
                if not _looks_like_function_signature(segment):
                    continue
                signatures.add(f"at::symint::{normalize_signature(segment)}")
    return signatures


def extract_member_definitions(header: Path) -> dict[str, set[str]]:
    text = _prepared_header_text(header)
    signatures: dict[str, set[str]] = {}
    for at_block in namespace_blocks(text, "at"):
        for kind, segment in split_top_level_declarations(at_block):
            if kind != "definition":
                continue
            if "Tensor::" not in segment:
                continue
            if not _looks_like_function_signature(segment):
                continue
            normalized = normalize_signature(segment)
            name = member_definition_name(normalized)
            if name:
                signatures.setdefault(name, set()).add(normalized)
    return signatures


def extract_tensor_class_members(header: Path) -> dict[str, set[str]]:
    text = _prepared_header_text(header)
    signatures: dict[str, set[str]] = {}
    for class_block in tensor_class_blocks(text):
        for _kind, segment in split_top_level_declarations(class_block):
            if not _looks_like_function_signature(segment):
                continue
            normalized = normalize_signature(segment)
            name = class_member_name(normalized)
            if name:
                signatures.setdefault(name, set()).add(normalized)
    return signatures


def read_torch_version(include_dir: Path) -> str | None:
    for version_rel in TORCH_VERSION_RELS:
        version_header = include_dir / version_rel
        if not version_header.exists():
            continue
        text = version_header.read_text()
        match = re.search(
            r'#define\s+TORCH_VERSION(?:\s+|(?:\s*\\\s*[\r\n]+\s*))'
            r'"([^"]+)"',
            text,
        )
        if match:
            return match.group(1)

        macros: dict[str, str] = {}
        for key in ("MAJOR", "MINOR", "PATCH"):
            macro_match = re.search(
                rf"#define\s+TORCH_VERSION_{key}\s+([0-9]+)", text
            )
            if macro_match:
                macros[key] = macro_match.group(1)
        if len(macros) == 3:
            return f"{macros['MAJOR']}.{macros['MINOR']}.{macros['PATCH']}"
    return None


def _python_torch_include_dir() -> Path | None:
    code = (
        "import pathlib, torch; "
        "print(pathlib.Path(torch.__file__).resolve().parent / 'include')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    include_dir = Path(result.stdout.strip())
    return include_dir if include_dir.exists() else None


def resolve_libtorch_include_dir(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.libtorch_include_dir:
        candidates.append(Path(args.libtorch_include_dir))

    env_include = os.environ.get("LIBTORCH_INCLUDE_DIR")
    if env_include:
        candidates.append(Path(env_include))

    if args.libtorch_root:
        candidates.append(Path(args.libtorch_root) / "include")

    env_root = os.environ.get("LIBTORCH_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "include")

    python_include = _python_torch_include_dir()
    if python_include:
        candidates.append(python_include)

    for include_dir in candidates:
        if not (include_dir / ATEN_OPS_REL).is_dir():
            continue
        if not (include_dir / TENSOR_BODY_REL).exists():
            continue
        return include_dir.resolve()

    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Cannot find libtorch include directory with ATen/ops headers. "
        f"Checked:\n  {checked}"
    )


def _compare_actual_signatures(
    title: str, expected: set[str], actual: set[str]
) -> SignatureIssue | None:
    extra = tuple(sorted(actual - expected))
    if not extra:
        return None
    return SignatureIssue(title=title, missing=(), extra=extra)


def filter_actual_free_functions(
    expected: set[str], actual: set[str]
) -> set[str]:
    expected_names = {
        name for item in expected if (name := signature_name(item))
    }
    return {
        item
        for item in actual
        if (name := signature_name(item)) and name in expected_names
    }


def collect_signature_issues(
    paddle_root: Path, libtorch_include_dir: Path
) -> list[SignatureIssue]:
    issues: list[SignatureIssue] = []
    paddle_ops_dir = paddle_root / PADDLE_OPS_REL
    paddle_tensor_body = paddle_root / PADDLE_TENSOR_BODY_REL
    lib_ops_dir = libtorch_include_dir / ATEN_OPS_REL
    lib_tensor_body = libtorch_include_dir / TENSOR_BODY_REL

    if not paddle_ops_dir.is_dir():
        raise FileNotFoundError(
            f"Paddle ops header directory not found: {paddle_ops_dir}"
        )
    if not paddle_tensor_body.exists():
        raise FileNotFoundError(
            f"Paddle TensorBody header not found: {paddle_tensor_body}"
        )

    lib_tensor_decls = extract_tensor_class_members(lib_tensor_body)
    lib_tensor_defs = extract_member_definitions(lib_tensor_body)
    paddle_tensor_decls = extract_tensor_class_members(paddle_tensor_body)

    for paddle_header in sorted(paddle_ops_dir.glob("*.h")):
        lib_header = lib_ops_dir / paddle_header.name
        if not lib_header.exists():
            issues.append(
                SignatureIssue(
                    title=(
                        f"{paddle_header.name}: matching libtorch header "
                        "not found"
                    ),
                    missing=(),
                    extra=(str(paddle_header),),
                )
            )
            continue

        expected_free = extract_free_functions(lib_header)
        actual_free = filter_actual_free_functions(
            expected_free, extract_free_functions(paddle_header)
        )
        issue = _compare_actual_signatures(
            f"{paddle_header.name}: ATen free function signatures differ",
            expected_free,
            actual_free,
        )
        if issue:
            issues.append(issue)

        paddle_member_defs = extract_member_definitions(paddle_header)
        for method_name, actual_defs in sorted(paddle_member_defs.items()):
            expected_decls = lib_tensor_decls.get(method_name, set())
            expected_defs = set(lib_tensor_defs.get(method_name, set()))
            expected_defs.update(
                class_declaration_to_definition_signature(item, method_name)
                for item in expected_decls
            )
            if not expected_defs:
                issues.append(
                    SignatureIssue(
                        title=(
                            f"{paddle_header.name}: Tensor::{method_name} "
                            "is not present in libtorch TensorBody.h"
                        ),
                        missing=(),
                        extra=tuple(sorted(actual_defs)),
                    )
                )
                continue

            issue = _compare_actual_signatures(
                f"{paddle_header.name}: Tensor::{method_name} "
                "definition signatures differ",
                expected_defs,
                actual_defs,
            )
            if issue:
                issues.append(issue)

            actual_decls = paddle_tensor_decls.get(method_name, set())
            issue = _compare_actual_signatures(
                f"{paddle_header.name}: Tensor::{method_name} "
                "declaration signatures differ in compat/ATen/core/"
                "TensorBody.h",
                expected_decls,
                actual_decls,
            )
            if issue:
                issues.append(issue)

    return sorted(issues, key=lambda item: item.title)


def load_baseline(path: Path) -> list[SignatureIssue]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        raw_issues = data.get("issues", [])
    else:
        raw_issues = data
    return [
        SignatureIssue.from_json(item)
        for item in raw_issues
        if isinstance(item, dict)
    ]


def write_baseline(
    path: Path, issues: list[SignatureIssue], torch_version: str
) -> None:
    payload = {
        "torch_version": torch_version,
        "description": (
            "Baseline of known Paddle compat ATen ops signature differences "
            "against torch 2.9.1 CPU headers."
        ),
        "issues": [issue.to_json() for issue in issues],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _issue_key(issue: SignatureIssue) -> str:
    return json.dumps(issue.to_json(), sort_keys=True)


def _format_issue(issue: SignatureIssue, max_items: int = 20) -> list[str]:
    lines = [issue.title]
    if issue.missing:
        lines.append("  Expected signatures:")
        lines.extend(f"    - {item}" for item in issue.missing[:max_items])
        if len(issue.missing) > max_items:
            lines.append(f"    ... {len(issue.missing) - max_items} more")
    if issue.extra:
        lines.append("  Paddle signatures not found in libtorch:")
        lines.extend(f"    - {item}" for item in issue.extra[:max_items])
        if len(issue.extra) > max_items:
            lines.append(f"    ... {len(issue.extra) - max_items} more")
    return lines


def _format_issue_delta(
    title: str, issues: list[SignatureIssue], max_issues: int = 20
) -> list[str]:
    if not issues:
        return []
    lines = [title]
    for issue in issues[:max_issues]:
        lines.extend(_format_issue(issue))
    if len(issues) > max_issues:
        lines.append(f"... {len(issues) - max_issues} more issue groups")
    return lines


def compare_with_baseline(
    current: list[SignatureIssue], baseline: list[SignatureIssue]
) -> list[str]:
    current_by_key = {_issue_key(issue): issue for issue in current}
    baseline_by_key = {_issue_key(issue): issue for issue in baseline}

    new_keys = sorted(current_by_key.keys() - baseline_by_key.keys())
    resolved_keys = sorted(baseline_by_key.keys() - current_by_key.keys())
    new_issues = [current_by_key[key] for key in new_keys]
    resolved_issues = [baseline_by_key[key] for key in resolved_keys]

    messages = []
    messages.extend(
        _format_issue_delta(
            "New or changed Paddle signatures not found in libtorch:",
            new_issues,
        )
    )
    messages.extend(
        _format_issue_delta(
            "Baseline differences no longer present; update the baseline:",
            resolved_issues,
        )
    )
    return messages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Paddle compat ATen ops signatures with libtorch headers."
        )
    )
    parser.add_argument(
        "--paddle-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Paddle repository root.",
    )
    parser.add_argument(
        "--libtorch-root",
        help="Path to a libtorch root; include/ is expected below it.",
    )
    parser.add_argument(
        "--libtorch-include-dir",
        help="Path to libtorch or torch wheel include directory.",
    )
    parser.add_argument(
        "--expected-torch-version",
        default=DEFAULT_EXPECTED_TORCH_VERSION,
        help="Expected TORCH_VERSION from libtorch headers.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        type=Path,
        help="Known-differences baseline JSON file.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline with the current signature differences.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        libtorch_include_dir = resolve_libtorch_include_dir(args)
        torch_version = read_torch_version(libtorch_include_dir)
        if torch_version != args.expected_torch_version:
            print(
                "libtorch version mismatch: "
                f"expected {args.expected_torch_version}, "
                f"got {torch_version or 'unknown'} from {libtorch_include_dir}",
                file=sys.stderr,
            )
            return 1

        current = collect_signature_issues(
            args.paddle_root.resolve(), libtorch_include_dir
        )
        if args.update_baseline:
            write_baseline(args.baseline, current, torch_version)
            print(f"Updated {args.baseline} with {len(current)} issue groups.")
            return 0

        if not args.baseline.exists():
            print(f"Baseline file not found: {args.baseline}", file=sys.stderr)
            return 1
        baseline = load_baseline(args.baseline)
        messages = compare_with_baseline(current, baseline)
        if messages:
            print("libtorch ATen ops signature check failed:", file=sys.stderr)
            print("\n".join(messages), file=sys.stderr)
            return 1

        print(
            "libtorch ATen ops signature check passed "
            f"with {len(current)} known issue groups."
        )
        return 0
    except Exception as exc:
        print(
            f"libtorch ATen ops signature check failed: {exc}", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
