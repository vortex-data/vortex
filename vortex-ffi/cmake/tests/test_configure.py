# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Configure real entrypoints and record Cargo handoffs without compiling Rust."""

import json
import os
import subprocess
import tomllib
import unittest
from pathlib import Path

from support import CMakeTest


class ConfigureTests(CMakeTest):
    def setUp(self) -> None:
        super().setUp()
        for name in ("CMAKE_BUILD_TYPE", "RUSTUP_TOOLCHAIN"):
            self.env.pop(name, None)
        self.env["CARGO_NET_OFFLINE"] = "true"
        self.cargo = self.recording_cargo()
        self.rustc = self.fake_rustc()
        # Defaults are directory-local variables, not necessarily cache entries.
        self.hook = self.write(
            "record-defaults.cmake",
            r"""
            function(record_defaults)
                file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/defaults.txt"
                    "${CMAKE_BUILD_TYPE}\n${VORTEX_WARNINGS_AS_ERRORS}\n")
                if(PROJECT_NAME STREQUAL "VortexCXX")
                    get_target_property(options vortex_cxx COMPILE_OPTIONS)
                    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/defaults.txt" "${options}\n")
                endif()
            endfunction()
            cmake_language(DEFER CALL record_defaults)
            """,
        )

    def configure(
        self, name: str, *options: str, source: Path | None = None, success: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.cmake_configure(
            source or self.repo / "vortex-ffi",
            self.work / name,
            f"-DVORTEX_CARGO_EXECUTABLE={self.cargo}",
            f"-DVORTEX_RUSTC_EXECUTABLE={self.rustc}",
            f"-DCMAKE_PROJECT_INCLUDE={self.hook}",
            "-DVORTEX_BUILD_TESTS=OFF",
            "-DVORTEX_BUILD_EXAMPLES=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            *options,
            success=success,
        )

    def test_standalone_defaults_and_ffi_default_build(self) -> None:
        for name, source, directories in (
            ("root", ".", ("ffi", "cpp")),
            ("ffi", "vortex-ffi", (".",)),
            ("cpp", "lang/cpp", ("ffi", ".")),
        ):
            with self.subTest(entrypoint=name):
                self.configure(name, source=self.repo / source)
                build = self.work / name
                self.assertFalse(list(build.rglob("libvortex_ffi.a")), "Configure must not run Cargo")
                for directory in directories:
                    settings = (build / directory / "defaults.txt").read_text().splitlines()
                    self.assertEqual(settings[:2], ["Debug", "ON"])
                    if len(settings) == 3:
                        self.assertIn("-Werror", settings[2].split(";"))
                        self.assertIn("$<$<COMPILE_LANGUAGE:C,CXX>:-g2>", settings[2].split(";"))

        build = self.work / "ffi"
        self.cmake_build(build)
        self.assertEqual((build / "vortex-artifacts/libvortex_ffi.a").read_bytes(), b"recorded archive")
        recorded = self.cargo_recording(build / "cargo-target")
        self.assertEqual(recorded["args"][recorded["args"].index("--profile") + 1], "dev")
        config = tomllib.loads((self.repo / ".cargo/config.toml").read_text())
        expected = config["target"]['cfg(target_family="unix")']["rustflags"] + ["-C", "relocation-model=pic"]
        expected += ["-C", "debuginfo=2"]
        self.assertEqual(recorded["env"]["CARGO_ENCODED_RUSTFLAGS"].split("\x1f"), expected)

        self.configure("ffi", "-DVORTEX_DEBUG_INFO=0")
        self.cmake_build(build)
        recorded = self.cargo_recording(build / "cargo-target")
        self.assertEqual(recorded["env"]["CARGO_ENCODED_RUSTFLAGS"].split("\x1f"), expected[:-1] + ["debuginfo=0"])

    def test_embedded_root_preserves_parent_variables(self) -> None:
        source = self.write(
            "parent/CMakeLists.txt",
            f"""\
            cmake_minimum_required(VERSION 3.25)
            project(Parent LANGUAGES C CXX)
            set(CMAKE_BUILD_TYPE "")
            set(_vortex_top_level ON)
            add_subdirectory("{self.repo}" vortex EXCLUDE_FROM_ALL)
            if(NOT CMAKE_BUILD_TYPE STREQUAL "" OR NOT _vortex_top_level STREQUAL "ON")
                message(FATAL_ERROR "Vortex changed parent variables")
            endif()
            """,
        ).parent
        self.configure("embedded", source=source)
        build = self.work / "embedded"
        for directory in ("ffi", "cpp"):
            settings = (build / "vortex" / directory / "defaults.txt").read_text().splitlines()
            self.assertEqual(settings[:2], ["", "OFF"])
            if directory == "cpp":
                self.assertNotIn("-Werror", settings[2].split(";"))
        self.cmake_build(build)
        self.assertFalse(list(build.rglob("libvortex_ffi.a")))

    def test_unused_embedded_ffi_keeps_cargo_lazy(self) -> None:
        source = self.write(
            "parent/CMakeLists.txt",
            f"""\
            cmake_minimum_required(VERSION 3.25)
            project(Parent LANGUAGES C CXX)
            add_subdirectory("{self.repo}/vortex-ffi" ffi)
            """,
        ).parent
        self.configure("lazy", source=source)
        build = self.work / "lazy"
        self.cmake_build(build)
        self.assertFalse(list(build.rglob("libvortex_ffi.a")))
        self.cmake_build(build, "--target", "vortex_ffi_cargo_build")
        self.assertEqual((build / "ffi/vortex-artifacts/libvortex_ffi.a").read_bytes(), b"recorded archive")

    def test_profile_mapping_and_override(self) -> None:
        for config, override, expected in (
            ("Release", "", "release"),
            ("RelWithDebInfo", "", "release_debug"),
            ("Debug", "ci", "ci"),
        ):
            with self.subTest(config=config, override=override):
                self.configure(config, f"-DCMAKE_BUILD_TYPE={config}", f"-DVORTEX_CARGO_PROFILE={override}")
                build = self.work / config
                self.cmake_build(build)
                args = self.cargo_recording(build / "cargo-target")["args"]
                self.assertEqual(args[args.index("--profile") + 1], expected)
                self.assertEqual((build / "vortex-artifacts/libvortex_ffi.a").read_bytes(), b"recorded archive")

    def test_sanitizer_rejections(self) -> None:
        result = self.configure("unknown", "-DVORTEX_SANITIZER=typo", success=False)
        self.assertIn("got 'typo'", result.stdout + result.stderr)
        for compiler, message in (
            ("AppleClang", "Rust sanitizer builds require upstream LLVM Clang"),
            ("Clang", "Rust sanitizer builds require nightly rustc"),
        ):
            with self.subTest(compiler=compiler):
                # Only override compiler identity; no sanitizer code is compiled.
                hook = self.write(
                    "compiler-ids.cmake",
                    f"set(CMAKE_C_COMPILER_ID {compiler})\nset(CMAKE_CXX_COMPILER_ID {compiler})\n",
                )
                result = self.configure(
                    compiler,
                    f"-DCMAKE_PROJECT_VortexFFI_INCLUDE={hook}",
                    "-DVORTEX_SANITIZER=asan",
                    success=False,
                )
                self.assertIn(message, result.stdout + result.stderr)

    def test_toolchain_selection_survives_reconfigure_and_explicit_updates(self) -> None:
        rustc_log = self.work / "rustc.json"
        build = self.work / "toolchain"

        def assert_selection(selected: str | None) -> None:
            self.assertEqual(json.loads(rustc_log.read_text()), [selected, str(self.repo)])
            recorded = self.cargo_recording(build / "cargo-target")
            self.assertEqual(recorded["env"].get("RUSTUP_TOOLCHAIN"), selected)
            self.assertIn(
                f"VORTEX_RUSTUP_TOOLCHAIN:STRING={selected or ''}",
                (build / "CMakeCache.txt").read_text().splitlines(),
            )
            rustc_log.unlink()
            (build / "cargo-target/environment.json").unlink()

        first, second = "nightly-2026-04-01", "nightly-2026-05-01"
        self.env["RUSTUP_TOOLCHAIN"] = first
        self.configure("toolchain")
        self.env.pop("RUSTUP_TOOLCHAIN")
        self.cmake_build(build)
        assert_selection(first)

        for ambient in (None, "ambient"):
            self.env.pop("RUSTUP_TOOLCHAIN", None)
            if ambient:
                self.env["RUSTUP_TOOLCHAIN"] = ambient
            # Make the manifest stale without clock-dependent waits or future-dated inputs.
            os.utime(build / "build.ninja", (1, 1))
            self.cmake_build(build)
            assert_selection(first)

        for selected in (second, None):
            with self.subTest(toolchain=selected):
                self.configure("toolchain", f"-DVORTEX_RUSTUP_TOOLCHAIN={selected or ''}")
                self.cmake_build(build)
                assert_selection(selected)

        os.utime(build / "build.ninja", (1, 1))
        self.cmake_build(build)
        assert_selection(None)


if __name__ == "__main__":
    unittest.main()
