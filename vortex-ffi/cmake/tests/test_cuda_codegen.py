# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Exercise the Rust CUDA consumers with real std-only bodies and recording NVCC."""

import json
import os
import unittest
from pathlib import Path

from support import CMakeTest, rust_toolchain_environment


def production_function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    body = source[source.index(f"\nfn {name}(") + 1 :].split("\nfn ", 1)[0]
    # Drop any doc comments belonging to the next top-level function.
    return body[: body.rindex("\n}") + 2] + "\n"


class CudaCodegenTests(CMakeTest):
    def setUp(self) -> None:
        super().setUp()
        self.output = self.work / "output's with spaces"
        self.output.mkdir()
        self.write("kernel.cu", "// Recording NVCC does not compile CUDA.\n")
        self.env = {
            name: value
            for name, value in rust_toolchain_environment().items()
            if not name.startswith("NVCC_") and name not in ("VORTEX_CUDA_ARCH_FLAGS", "VORTEX_CUDA_HOST_COMPILER")
        }
        self.env.update(PATH=str(self.work) + os.pathsep + self.env.get("PATH", ""), PROFILE="release")
        self.executable(
            "nvcc",
            """\
            import json, os, sys
            from pathlib import Path
            args = sys.argv[1:]
            environment = {
                name: os.environ.get(name)
                for name in ("NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
            }
            with Path(__file__).with_name("nvcc.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"args": args, "env": environment}) + "\\n")
            Path(args[args.index("-o") + 1]).write_bytes(b"\\x00\\xff" + json.dumps(args).encode())
            """,
        )
        cuda = self.repo / "vortex-cuda"
        functions = "\n".join(
            production_function(cuda / path, name)
            for path, name in (
                ("build.rs", "nvcc_compile_kernel"),
                ("build.rs", "generate_embedded_kernels"),
                ("cub/build.rs", "compile_shared_library"),
            )
        )
        source = self.write(
            "harness.rs",
            "use std::{env, fs::File, io::{self, Write}, path::{Path, PathBuf}, process::Command};\n"
            + functions
            + """
            fn main() -> io::Result<()> {
                let root = PathBuf::from(env::args_os().nth(1).unwrap());
                let output = root.join("output's with spaces");
                if env::args().nth(2).as_deref() == Some("empty") {
                    return generate_embedded_kernels(&output, &[]);
                }
                let source = root.join("kernel.cu");
                let fatbin = nvcc_compile_kernel(&root, &output, &source, "release")?;
                assert_eq!(fatbin, output.join("kernel.fatbin"));
                compile_shared_library(&root, &[source], &output);
                generate_embedded_kernels(&output, &[fatbin])
            }
            """,
        )
        self.harness = self.work / "harness"
        self.command("rustc", "--edition=2024", source, "-o", self.harness)

    def test_architecture_flags_modes_and_reused_outputs(self) -> None:
        explicit = [
            "--generate-code=arch=compute_80,code=[compute_80,sm_80]",
            "--generate-code=arch=compute_90,code=[compute_90]",
        ]
        outputs = (("--fatbin", "kernel.fatbin"), ("--shared", "libvortex_cub.so"))
        ambient = {
            "NVCC_CCBIN": "default-g++",
            "NVCC_PREPEND_FLAGS": "-ccbin=prepend-g++",
            "NVCC_APPEND_FLAGS": "-ccbin=append-g++ --use_fast_math",
        }
        compiler = os.fsdecode(os.fsencode(self.work) + b"/parent toolchain's non-utf8-\xff/g++")
        # An explicit empty value is CMake OFF's consumer-side representation.
        for flags, expected, host_compiler in (
            (None, ["-arch=native"], None),
            ("", [], ""),
            (" ".join(explicit), explicit, str(self.work / "parent toolchain's/g++")),
            (" \t" + "\n".join(explicit) + "\n", explicit, compiler),
        ):
            with self.subTest(flags=flags, host_compiler=host_compiler):
                env = self.env | ambient
                if flags is not None:
                    env["VORTEX_CUDA_ARCH_FLAGS"] = flags
                if host_compiler is not None:
                    env["VORTEX_CUDA_HOST_COMPILER"] = host_compiler
                log = self.work / "nvcc.jsonl"
                log.unlink(missing_ok=True)
                self.command(self.harness, self.work, env=env)
                calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(calls), 2)
                for call, (mode, name) in zip(calls, outputs, strict=True):
                    args = call["args"]
                    self.assertEqual(call["env"], ambient)
                    self.assertEqual(args.count("--compiler-bindir"), 1 if host_compiler else 0)
                    if host_compiler:
                        self.assertEqual(args[args.index("--compiler-bindir") + 1], host_compiler)
                    self.assertEqual([arg for arg in args if arg.startswith(("-arch", "--generate-code"))], expected)
                    self.assertNotIn("", args)
                    self.assertEqual([arg for arg in args if arg in ("--fatbin", "--shared", "--ptx")], [mode])
                    self.assertEqual(Path(args[args.index("-o") + 1]), self.output / name)
                    # Reused outputs must contain this invocation, not the previous architecture flags.
                    self.assertEqual((self.output / name).read_bytes(), b"\x00\xff" + json.dumps(args).encode())

    def test_binary_embedding_and_empty_table_exclude_stale_fatbins(self) -> None:
        stale = self.output / "stale.fatbin"
        stale.write_bytes(b"\x00\xfe")
        for empty in (False, True):
            with self.subTest(empty=empty):
                self.command(self.harness, self.work, "empty" if empty else "compile")
                fatbin = (self.output / "kernel.fatbin").read_bytes()
                expected = "None" if empty else f"Some(&{list(fatbin)}[..])"
                source = self.write(
                    "consumer.rs",
                    f"""\
                    include!(r#"{self.output.as_posix()}/embedded_kernels.rs"#);
                    fn main() {{
                        assert_eq!(embedded_kernel("kernel"), {expected});
                        assert_eq!(embedded_kernel("stale"), None);
                        assert_eq!(embedded_kernel("missing"), None);
                    }}
                    """,
                )
                consumer = self.work / "consumer"
                self.command("rustc", "--edition=2024", source, "-o", consumer)
                self.command(consumer)
                self.assertEqual(stale.read_bytes(), b"\x00\xfe")


if __name__ == "__main__":
    unittest.main()
