# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline CMake regression tests; run with python3 -B path/to/test_build_integration.py."""

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = Path(__file__).with_name("upstream.patch")
MODULE = Path("cpp/benchmarks/ndsh/vortex.cmake")
SMOKE = MODULE.with_name("vortex_build_smoke.cpp")
IO = MODULE.with_name("vortex_io.cpp")
IO_TEST = MODULE.with_name("vortex_io_test.cpp")
CUDA_HEADER = Path("vortex-cuda/ffi/cinclude/vortex_cuda.h")

PARENT = """
cmake_minimum_required(VERSION 3.28)
project(ndsh_integration_test LANGUAGES CXX)

function(assert_equal actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "Expected '${expected}', got '${actual}' (${ARGN})")
  endif()
endfunction()

function(assert_property target property expected)
  get_target_property(actual ${target} ${property})
  if(actual STREQUAL "actual-NOTFOUND")
    set(actual "")
  endif()
  assert_equal("${actual}" "${expected}" "${target}.${property}")
endfunction()

set(VORTEX_ENABLE_CUDA OFF CACHE BOOL "Parent CUDA policy")
set(VORTEX_BUILD_TESTS ON CACHE BOOL "Parent test policy")
set(VORTEX_BUILD_EXAMPLES ON CACHE BOOL "Parent example policy")
set(VORTEX_WARNINGS_AS_ERRORS ON CACHE BOOL "Parent warning policy")
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "Parent architecture")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON CACHE BOOL "Parent warning policy")
set(CMAKE_CXX_FLAGS "-DPARENT_CXX_FLAG=1" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-DPARENT_DEBUG_FLAG=1" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CUDA_FLAGS "--parent-cuda-flag" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG "--parent-cuda-debug-flag" CACHE STRING "Parent flags" FORCE)
get_cmake_property(parent_variables VARIABLES)
list(FILTER parent_variables INCLUDE REGEX "^(VORTEX_|CMAKE_.*(FLAGS|ARCHITECTURES|STANDARD|WARNING_AS_ERROR))")
foreach(variable IN LISTS parent_variables)
  set(before_${variable} "${${variable}}")
  foreach(property IN ITEMS VALUE TYPE HELPSTRING)
    get_property(before_${variable}_${property} CACHE ${variable} PROPERTY ${property})
  endforeach()
endforeach()

add_library(cudf INTERFACE)
add_library(cudf::cudf INTERFACE IMPORTED)
add_library(nanoarrow::nanoarrow INTERFACE IMPORTED)
add_library(CUDA::cudart INTERFACE IMPORTED)
target_compile_definitions(nanoarrow::nanoarrow INTERFACE NANOARROW_FAKE_LINK=1)
target_compile_definitions(CUDA::cudart INTERFACE CUDA_FAKE_LINK=1)
add_library(ndsh_utilities INTERFACE)
set(benchmarks NDSH_Q01_NVBENCH NDSH_Q06_NVBENCH NDSH_Q05_NVBENCH
               NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH NDSH_HELPER)
foreach(target IN LISTS benchmarks)
  add_executable(${target} EXCLUDE_FROM_ALL dummy.cpp)
  target_link_libraries(${target} PRIVATE cudf::cudf ndsh_utilities)
endforeach()
add_executable(unrelated dummy.cpp)

include(cpp/benchmarks/ndsh/vortex.cmake)

foreach(variable IN LISTS parent_variables)
  assert_equal("${${variable}}" "${before_${variable}}" "parent ${variable}")
  foreach(property IN ITEMS VALUE TYPE HELPSTRING)
    get_property(after CACHE ${variable} PROPERTY ${property})
    assert_equal("${after}" "${before_${variable}_${property}}" "cache ${variable}.${property}")
  endforeach()
endforeach()

foreach(target IN LISTS benchmarks)
  set(links "cudf::cudf;ndsh_utilities")
  set(definitions "")
  if(CUDF_NDSH_WITH_VORTEX AND target MATCHES "^NDSH_Q0[16]_NVBENCH$")
    list(APPEND links NDSH_VORTEX_IO)
    set(definitions CUDF_NDSH_WITH_VORTEX=1)
  endif()
  assert_property(${target} LINK_LIBRARIES "${links}")
  assert_property(${target} COMPILE_DEFINITIONS "${definitions}")
  assert_property(${target} INTERFACE_LINK_LIBRARIES "")
  assert_property(${target} INTERFACE_COMPILE_DEFINITIONS "")
  if(definitions)
    target_compile_definitions(${target} PRIVATE EXPECT_VORTEX=1)
  endif()
endforeach()
foreach(target IN ITEMS cudf cudf::cudf ndsh_utilities unrelated)
  foreach(property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES
                            COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
    assert_property(${target} ${property} "")
  endforeach()
endforeach()

foreach(target IN ITEMS nanoarrow::nanoarrow CUDA::cudart)
  foreach(property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS)
    assert_property(${target} ${property} "")
  endforeach()
endforeach()
assert_property(nanoarrow::nanoarrow INTERFACE_COMPILE_DEFINITIONS NANOARROW_FAKE_LINK=1)
assert_property(CUDA::cudart INTERFACE_COMPILE_DEFINITIONS CUDA_FAKE_LINK=1)

set(integration_targets NDSH_VORTEX_IO NDSH_VORTEX_BUILD_SMOKE NDSH_VORTEX_IO_TEST)
if(CUDF_NDSH_WITH_VORTEX)
  foreach(target IN ITEMS Vortex::cpp_static ${integration_targets})
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Missing enabled Vortex target: ${target}")
    endif()
  endforeach()
  foreach(target IN LISTS integration_targets)
    assert_property(${target} EXCLUDE_FROM_ALL TRUE)
    assert_property(${target} COMPILE_DEFINITIONS "")
    assert_property(${target} INTERFACE_COMPILE_DEFINITIONS "")
  endforeach()
  assert_property(NDSH_VORTEX_IO TYPE STATIC_LIBRARY)
  assert_property(NDSH_VORTEX_IO LINK_LIBRARIES "cudf::cudf;Vortex::cpp_static;nanoarrow::nanoarrow;CUDA::cudart")
  assert_property(NDSH_VORTEX_IO INTERFACE_LINK_LIBRARIES
                  "cudf::cudf;$<LINK_ONLY:Vortex::cpp_static>;$<LINK_ONLY:nanoarrow::nanoarrow>;$<LINK_ONLY:CUDA::cudart>")
  assert_property(NDSH_VORTEX_IO COMPILE_FEATURES cxx_std_20)
  assert_property(NDSH_VORTEX_IO INTERFACE_COMPILE_FEATURES cxx_std_20)
  assert_property(NDSH_VORTEX_IO INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/cpp/benchmarks/ndsh")
  assert_property(NDSH_VORTEX_IO INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/cpp/benchmarks/ndsh")
  assert_property(NDSH_VORTEX_BUILD_SMOKE LINK_LIBRARIES "cudf::cudf;Vortex::cpp_static")
  assert_property(NDSH_VORTEX_BUILD_SMOKE INTERFACE_LINK_LIBRARIES "")
  assert_property(NDSH_VORTEX_BUILD_SMOKE COMPILE_FEATURES cxx_std_20)
  assert_property(NDSH_VORTEX_IO_TEST LINK_LIBRARIES "NDSH_VORTEX_IO;nanoarrow::nanoarrow;CUDA::cudart")
  assert_property(NDSH_VORTEX_IO_TEST INTERFACE_LINK_LIBRARIES "")
else()
  foreach(target IN ITEMS Vortex::cpp_static vortex_unwanted ${integration_targets})
    if(TARGET ${target})
      message(FATAL_ERROR "Disabled integration created Vortex target: ${target}")
    endif()
  endforeach()
endif()
"""

FAKE_VORTEX = """
project(fake_vortex LANGUAGES CXX)
assert_equal("${VORTEX_ENABLE_CUDA}" ON)
assert_equal("${VORTEX_BUILD_TESTS}" OFF)
assert_equal("${VORTEX_BUILD_EXAMPLES}" OFF)
assert_equal("${VORTEX_WARNINGS_AS_ERRORS}" OFF)
assert_equal("${CMAKE_CUDA_ARCHITECTURES}" 90)
assert_equal("${CMAKE_CXX_FLAGS}" "-DPARENT_CXX_FLAG=1")
assert_equal("${CMAKE_CUDA_FLAGS}" "--parent-cuda-flag")
add_library(vortex_cpp_static INTERFACE)
add_library(Vortex::cpp_static ALIAS vortex_cpp_static)
target_compile_definitions(vortex_cpp_static INTERFACE VORTEX_FAKE_LINK=1)
# A default build must not enter the FetchContent dependency's ALL target.
add_custom_target(vortex_unwanted ALL COMMAND "${CMAKE_COMMAND}" -E false)
"""

DUMMY = """
#ifndef PARENT_CXX_FLAG
#error Parent compiler flags were lost
#endif
#if defined(VORTEX_FAKE_LINK) || defined(NANOARROW_FAKE_LINK) || defined(CUDA_FAKE_LINK)
#error Private adapter dependency usage requirements leaked
#endif
#ifdef EXPECT_VORTEX
#if !defined(CUDF_NDSH_WITH_VORTEX) || CUDF_NDSH_WITH_VORTEX != 1
#error Missing private benchmark Vortex definition
#endif
int ndsh_vortex_io_stub();
int main() { return ndsh_vortex_io_stub(); }
#else
#ifdef CUDF_NDSH_WITH_VORTEX
#error Benchmark Vortex definition leaked
#endif
int main() { return 0; }
#endif
"""

IO_STUB = """
#ifndef PARENT_CXX_FLAG
#error Parent compiler flags were lost
#endif
#if !defined(VORTEX_FAKE_LINK) || !defined(NANOARROW_FAKE_LINK) || !defined(CUDA_FAKE_LINK)
#error Missing private adapter dependency usage requirements
#endif
#if defined(CUDF_NDSH_WITH_VORTEX) || defined(EXPECT_VORTEX)
#error Benchmark definitions leaked into the adapter
#endif
int ndsh_vortex_io_stub() { return 0; }
"""


class BuildIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool in ("git", "cmake", "c++"):
            if shutil.which(tool) is None:
                raise unittest.SkipTest(f"{tool} is required")
        version = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.split()[2]
        if tuple(map(int, version.split(".")[:2])) < (3, 28):
            raise unittest.SkipTest("CMake >= 3.28 is required for FetchContent EXCLUDE_FROM_ALL")

    def setUp(self):
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="ndsh-unittest-", dir=build)
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)
        self.binary = self.source / "out"
        self.fake = self.source / "fake-vortex"
        # A separate repository prevents git apply from treating this as a workspace subdirectory.
        self.run_command("git", "init", "--quiet")
        files = (MODULE, SMOKE, IO, IO.with_suffix(".hpp"), IO_TEST)
        includes = [argument for path in files for argument in ("--include", str(path))]
        self.run_command("git", "apply", *includes, str(PATCH))
        for path in files:
            self.assertTrue((self.source / path).is_file(), f"Patch is missing {path}")
        # Compile tests cover wiring with a stub; the real adapter needs separate GPU validation.
        # Leave the smoke/I/O test sources intact: neither may build implicitly without CUDA/Rust.
        self.write(IO, IO_STUB)
        self.write("CMakeLists.txt", PARENT)
        self.write("dummy.cpp", DUMMY)
        self.write("fake-vortex/CMakeLists.txt", 'message(FATAL_ERROR "Expected SOURCE_SUBDIR lang/cpp")')
        self.write("fake-vortex/lang/cpp/CMakeLists.txt", FAKE_VORTEX)
        self.write(
            Path("fake-vortex") / CUDA_HEADER,
            "#define VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES (1U << 1)\n"
            "int vx_cuda_scan_path_arrow_device_stream_projected(void);\n",
        )

    def write(self, path, content):
        destination = self.source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(textwrap.dedent(content), encoding="utf-8")

    def run_command(self, *command, succeeds=True):
        result = subprocess.run(
            command, cwd=self.source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, f"{command}\n{result.stdout}")
        else:
            self.assertNotEqual(result.returncode, 0, f"{command} unexpectedly succeeded\n{result.stdout}")
        return result.stdout

    def configure(self, *options, local_source=True, succeeds=True):
        source_override = (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={self.fake}",) if local_source else ()
        return self.run_command(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(self.binary),
            # Exercise the module's Linux gate without enabling CUDA, even on a non-Linux host.
            "-DCMAKE_SYSTEM_NAME=Linux",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            *source_override,
            *options,
            succeeds=succeeds,
        )

    def build(self, *targets):
        options = ("--target", *targets) if targets else ()
        self.run_command("cmake", "--build", str(self.binary), "--parallel", "2", *options)

    def test_disabled_by_default_and_explicitly(self):
        self.write("fake-vortex/lang/cpp/CMakeLists.txt", 'message(FATAL_ERROR "Disabled Vortex was configured")')
        for setting in (None, "OFF"):
            with self.subTest(setting=setting):
                self.binary = self.source / (setting or "default")
                self.configure(*(() if setting is None else (f"-DCUDF_NDSH_WITH_VORTEX={setting}",)))
                cache = (self.binary / "CMakeCache.txt").read_text(encoding="utf-8")
                self.assertIn("CUDF_NDSH_WITH_VORTEX:BOOL=OFF", cache)
                self.build()
                self.build("NDSH_Q01_NVBENCH", "NDSH_Q06_NVBENCH")

    def test_enabled_private_links_and_excluded_targets(self):
        self.configure("-DCUDF_NDSH_WITH_VORTEX=ON")
        self.build("unrelated")
        # Also check ALL: explicitly building unrelated alone cannot detect a missing exclusion.
        self.build()
        archive = self.binary / "libNDSH_VORTEX_IO.a"
        self.assertFalse(archive.exists(), "Default build included the adapter")
        self.build("NDSH_Q05_NVBENCH", "NDSH_Q09_NVBENCH", "NDSH_Q10_NVBENCH", "NDSH_HELPER")
        self.assertFalse(archive.exists(), "Unselected benchmarks built the adapter")
        self.build("NDSH_Q01_NVBENCH", "NDSH_Q06_NVBENCH")
        self.assertTrue(archive.is_file(), "Selected benchmarks did not build the static adapter")
        for target in ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST"):
            self.assertFalse((self.binary / target).exists(), f"Explicit target {target} was built implicitly")

    def test_disabled_ignores_missing_or_stale_local_source(self):
        (self.fake / CUDA_HEADER).unlink()
        for setting in (None, "OFF"):
            for index, override in enumerate((None, "", self.fake, self.source / "missing-vortex")):
                with self.subTest(setting=setting, override=override):
                    self.binary = self.source / f"disabled-{setting}-{index}"
                    self.configure(
                        *(() if setting is None else (f"-DCUDF_NDSH_WITH_VORTEX={setting}",)),
                        *(() if override is None else (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={override}",)),
                        local_source=False,
                    )

    def test_missing_local_source_fails_offline(self):
        for index, override in enumerate((None, "")):
            with self.subTest(override=override):
                self.binary = self.source / f"missing-{index}"
                output = self.configure(
                    "-DCUDF_NDSH_WITH_VORTEX=ON",
                    *(() if override is None else (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={override}",)),
                    local_source=False,
                    succeeds=False,
                )
                message = " ".join(output.split())
                self.assertIn("local Vortex checkout with prerequisite fixes", message)
                self.assertIn("-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex", message)
                self.assertIn("published immutable revision", message)

    def test_stale_local_header_fails_offline(self):
        # Do not inspect Rust implementation details: only require the public FFI capability marker.
        headers = (
            None,
            "#pragma once\n",
            "// VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES is not available\n",
            "#define VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES_OLD (1U << 1)\n",
        )
        for index, header in enumerate(headers):
            with self.subTest(header=header):
                self.binary = self.source / f"stale-{index}"
                if header is None:
                    (self.fake / CUDA_HEADER).unlink()
                else:
                    self.write(Path("fake-vortex") / CUDA_HEADER, header)
                output = self.configure("-DCUDF_NDSH_WITH_VORTEX=ON", succeeds=False)
                message = " ".join(output.split())
                self.assertIn("VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES", message)
                self.assertIn(str(self.fake / CUDA_HEADER), output)
                self.assertIn("local Vortex checkout with prerequisite fixes", message)

    def test_local_header_missing_projection_fails_offline(self):
        self.write(Path("fake-vortex") / CUDA_HEADER, "#define VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES (1U << 1)\n")
        output = self.configure("-DCUDF_NDSH_WITH_VORTEX=ON", succeeds=False)
        message = " ".join(output.split())
        self.assertIn("vx_cuda_scan_path_arrow_device_stream_projected", message)
        self.assertIn(str(self.fake / CUDA_HEADER), output)
        self.assertIn("local Vortex checkout with prerequisite fixes", message)
        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex", message)

    def test_invalid_local_source_fails_offline(self):
        missing = self.source / "missing-vortex"
        output = self.configure(
            "-DCUDF_NDSH_WITH_VORTEX=ON", f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={missing}", succeeds=False
        )
        self.assertIn("FETCHCONTENT_SOURCE_DIR_VORTEX", output)
        self.assertIn(str(missing), output)

    def test_patch_hook_and_immutable_pin(self):
        patch = PATCH.read_text(encoding="utf-8")
        self.assertIn(
            " ConfigureNVBench(NDSH_Q10_NVBENCH ndsh/q10.cpp ndsh/utilities.cpp)\n+include(ndsh/vortex.cmake)\n",
            patch,
        )
        module = (self.source / MODULE).read_text(encoding="utf-8")
        self.assertRegex(module, r"(?m)^\s*GIT_TAG bffdca1109e99e6957ea2fc18f4a7809c88e0a0c\s*$")
        self.assertRegex(module, r"(?m)^\s*GIT_REPOSITORY https://github.com/vortex-data/vortex\.git\s*$")
        self.assertRegex(module, r"(?m)^\s*GIT_SHALLOW FALSE\s*$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
