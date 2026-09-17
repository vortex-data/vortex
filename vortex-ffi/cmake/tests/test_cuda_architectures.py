# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Exercise CUDA policy and Cargo forwarding without a CUDA toolkit."""

import shutil
import unittest
from pathlib import Path

from support import CMAKE_DIR, CMakeTest

ARCHITECTURES = "80;86-real;90a-virtual;100f"
ARCH_FLAGS = [
    "--generate-code=arch=compute_80,code=[compute_80,sm_80]",
    "--generate-code=arch=compute_86,code=[sm_86]",
    "--generate-code=arch=compute_90a,code=[compute_90a]",
    "--generate-code=arch=compute_100f,code=[compute_100f,sm_100f]",
]


class CudaArchitectureTests(CMakeTest):
    def test_policy_preserves_parent(self) -> None:
        for architectures, expected in (
            (None, "-arch=native"),
            ("native", "-arch=native"),
            ("OFF", ""),
            (ARCHITECTURES, ";".join(ARCH_FLAGS)),
            ("", None),
            ("80;;90", None),
            ("native;80", None),
            ("sm_80", None),
        ):
            with self.subTest(architectures=architectures):
                setup, parent_check = "", "NOT DEFINED CMAKE_CUDA_ARCHITECTURES"
                if architectures is not None:
                    setup = f"set(CMAKE_CUDA_ARCHITECTURES [==[{architectures}]==])"
                    parent_check = f"CMAKE_CUDA_ARCHITECTURES STREQUAL [==[{architectures}]==]"
                script = self.write(
                    "policy.cmake",
                    f"""\
                    cmake_minimum_required(VERSION 3.25)
                    include("{CMAKE_DIR}/Cuda.cmake")
                    {setup}
                    set(flags "stale output")
                    _vortex_resolve_cuda_architectures(flags)
                    if(NOT ({parent_check}) OR DEFINED CACHE{{CMAKE_CUDA_ARCHITECTURES}})
                        message(FATAL_ERROR "Parent policy changed")
                    endif()
                    file(WRITE flags.txt "${{flags}}")
                    """,
                )
                result = self.command("cmake", "-P", script, success=expected is not None)
                if expected is None:
                    self.assertIn("CMAKE_CUDA_ARCHITECTURES", result.stdout + result.stderr)
                else:
                    self.assertEqual((self.work / "flags.txt").read_text(encoding="utf-8"), expected)

    def cuda_fixture(self) -> tuple[Path, Path]:
        cuda_root = self.work / "fake CUDA toolkit's/targets/sbsa-linux"
        nvcc = self.executable("fake CUDA toolkit's/bin/nvcc", "raise SystemExit('No native compilation expected')\n")
        target_nvcc = cuda_root / "bin/nvcc"
        target_nvcc.parent.mkdir(parents=True)
        target_nvcc.symlink_to("../../../bin/nvcc")
        # Stub discovery only; production Configure.cmake and its Cargo driver run unchanged.
        self.write(
            "source/FindCUDAToolkit.cmake",
            f"""\
            if(NOT VORTEX_ENABLE_CUDA)
                message(FATAL_ERROR "CPU build must not discover CUDA")
            endif()
            set(CUDAToolkit_FOUND TRUE)
            set(CUDAToolkit_NVCC_EXECUTABLE "{target_nvcc}")
            set(CUDAToolkit_TARGET_DIR "{cuda_root}")
            """,
        )
        source = self.write(
            "source/CMakeLists.txt",
            f"""\
            cmake_minimum_required(VERSION 3.25)
            project(CudaArchitectureFixture LANGUAGES C)
            set(CMAKE_SYSTEM_NAME Linux)
            set(APPLE FALSE)
            set(CMAKE_BUILD_TYPE Debug)
            list(PREPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}")
            add_library(Threads::Threads INTERFACE IMPORTED)
            set(VORTEX_CARGO_EXECUTABLE "{self.recording_cargo()}")
            set(VORTEX_RUSTC_EXECUTABLE "{self.fake_rustc(host="x86_64-unknown-linux-gnu")}")
            foreach(tool IN ITEMS C_COMPILER CXX_COMPILER AR RANLIB)
                set(CMAKE_${{tool}} "{nvcc}")
            endforeach()
            if(DEFINED FIXTURE_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER "${{FIXTURE_CUDA_HOST_COMPILER}}")
            endif()
            include("{CMAKE_DIR}/Configure.cmake")
            if(DEFINED FIXTURE_CUDA_HOST_COMPILER AND
               NOT CMAKE_CUDA_HOST_COMPILER STREQUAL FIXTURE_CUDA_HOST_COMPILER)
                message(FATAL_ERROR "Parent CUDA host compiler changed")
            elseif(NOT DEFINED FIXTURE_CUDA_HOST_COMPILER AND DEFINED CMAKE_CUDA_HOST_COMPILER)
                message(FATAL_ERROR "Vortex selected a CUDA host compiler")
            endif()
            if(DEFINED CACHE{{CMAKE_CUDA_HOST_COMPILER}})
                message(FATAL_ERROR "Vortex cached a CUDA host compiler")
            endif()
            """,
        ).parent
        return source, cuda_root

    def test_configure_forwards_flags_and_cpu_ignores_policy(self) -> None:
        source, cuda_root = self.cuda_fixture()
        compiler = self.executable("parent toolchain's/bin/g++", "raise SystemExit('No compilation expected')\n")
        build = self.work / "build"
        for cuda, architectures, expected, selected in (
            ("ON", ARCHITECTURES, " ".join(ARCH_FLAGS), compiler),
            ("ON", "OFF", "", None),
            ("OFF", "invalid;native", None, "ccache;g++"),
        ):
            with self.subTest(cuda=cuda, architectures=architectures, selected=selected):
                env = self.env | {
                    "VORTEX_CUDA_HOST_COMPILER": "build-time-private-g++",
                    "NVCC_CCBIN": "build-time-g++",
                    "NVCC_PREPEND_FLAGS": "-ccbin=prepend-g++",
                    "NVCC_APPEND_FLAGS": "-ccbin=append-g++ --use_fast_math",
                }
                env.pop("CUDA_PATH", None)
                env.pop("VORTEX_CUDA_ARCH_FLAGS", None)
                if cuda == "ON":
                    env["VORTEX_CUDA_ARCH_FLAGS"] = "-arch=ambient"
                self.cmake_configure(
                    source,
                    build,
                    f"-DVORTEX_ENABLE_CUDA={cuda}",
                    f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
                    "-UFIXTURE_CUDA_HOST_COMPILER"
                    if selected is None
                    else f"-DFIXTURE_CUDA_HOST_COMPILER={selected}",
                    env=env | {
                        "NVCC_CCBIN": "configure-time-g++",
                        "VORTEX_CUDA_HOST_COMPILER": "configure-time-private-g++",
                    },
                )
                self.cmake_build(build, "--target", "vortex_ffi_cargo_build", env=env)
                self.assertEqual((build / "vortex-artifacts/libvortex_ffi.a").read_bytes(), b"recorded archive")
                recording = self.cargo_recording(build / "cargo-target")
                args, forwarded = recording["args"], recording["env"]
                self.assertEqual(args[args.index("--package") + 1], "vortex-cuda-ffi" if cuda == "ON" else "vortex-ffi")
                self.assertEqual(forwarded.get("VORTEX_CUDA_ARCH_FLAGS"), expected)
                self.assertEqual(forwarded.get("CUDA_PATH"), str(cuda_root) if cuda == "ON" else None)
                host_compiler = (selected or None) if cuda == "ON" else env["VORTEX_CUDA_HOST_COMPILER"]
                self.assertEqual(forwarded.get("VORTEX_CUDA_HOST_COMPILER"), host_compiler)
                for name in ("NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
                    self.assertEqual(forwarded.get(name), env.get(name))
                if cuda == "ON":
                    # Compare the invocation path: resolving the lookup result would hide the bug.
                    self.assertEqual(
                        shutil.which("nvcc", path=forwarded["PATH"]), str((cuda_root / "bin/nvcc").resolve())
                    )

    def test_host_compiler_rejects_command_arguments(self) -> None:
        source, _ = self.cuda_fixture()
        for compiler in ("ccache;g++", "g++ --sysroot=/sdk"):
            with self.subTest(compiler=compiler):
                result = self.cmake_configure(
                    source,
                    self.work / "invalid-host-compiler",
                    "-DVORTEX_ENABLE_CUDA=ON",
                    f"-DFIXTURE_CUDA_HOST_COMPILER={compiler}",
                    success=False,
                )
                self.assertIn(
                    "CMAKE_CUDA_HOST_COMPILER must name a single compiler executable", result.stdout + result.stderr
                )


if __name__ == "__main__":
    unittest.main()
