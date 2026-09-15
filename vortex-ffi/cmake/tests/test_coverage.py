# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Exercise the production coverage script without Rust, fetched dependencies, or lcov."""

import json
import os
import shutil
import unittest
from pathlib import Path

from support import CMakeTest


class CoverageTests(CMakeTest):
    def setUp(self) -> None:
        super().setUp()
        for name in list(self.env):
            if name.startswith(("CMAKE_", "GCOV_")) or name == "CXXFLAGS":
                self.env.pop(name)
        self.env.update(CMAKE_GENERATOR="Ninja", CMAKE_BUILD_PARALLEL_LEVEL="2")
        self.source = self.write(
            "cpp/CMakeLists.txt",
            """\
            # SPDX-License-Identifier: Apache-2.0
            # SPDX-FileCopyrightText: Copyright the Vortex contributors
            cmake_minimum_required(VERSION 3.25)
            project(CoverageFixture LANGUAGES CXX)
            option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
            option(VORTEX_BUILD_TESTS "Build tests" OFF)
            if(BUILD_SHARED_LIBS)
                set(coverage_target vortex_cxx_shared)
            else()
                set(coverage_target vortex_cxx_static)
            endif()
            add_library(${coverage_target} src/value.cpp)
            if(VORTEX_BUILD_TESTS)
                enable_testing()
                add_subdirectory(tests)
            endif()
            """,
        ).parent
        self.write(
            self.source / "src/value.cpp",
            """\
            // SPDX-License-Identifier: Apache-2.0
            // SPDX-FileCopyrightText: Copyright the Vortex contributors
            int value() { return 42; }
            """,
        )
        self.write(
            self.source / "tests/CMakeLists.txt",
            """\
            # SPDX-License-Identifier: Apache-2.0
            # SPDX-FileCopyrightText: Copyright the Vortex contributors
            add_executable(vortex_cxx_test main.cpp)
            target_link_libraries(vortex_cxx_test PRIVATE ${coverage_target})
            add_test(NAME coverage COMMAND vortex_cxx_test)
            """,
        )
        self.write(
            self.source / "tests/main.cpp",
            """\
            // SPDX-License-Identifier: Apache-2.0
            // SPDX-FileCopyrightText: Copyright the Vortex contributors
            int value();
            int main() { return value() == 42 ? 0 : 1; }
            """,
        )
        shutil.copyfile(self.repo / "lang/cpp/gcov-report.sh", self.source / "gcov-report.sh")
        self.launcher = self.executable(
            "reject-launcher", 'raise SystemExit("coverage compiler launcher must not run")\n'
        )
        self.executable(
            "bin/geninfo",
            """\
            # SPDX-License-Identifier: Apache-2.0
            # SPDX-FileCopyrightText: Copyright the Vortex contributors
            import json
            import sys
            from pathlib import Path
            Path("geninfo-args.json").write_text(json.dumps(sys.argv[1:]))
            """,
        )
        self.env["PATH"] = str(self.work / "bin") + os.pathsep + self.env["PATH"]

    def run_report(self) -> None:
        self.command("sh", self.source / "gcov-report.sh")
        args = json.loads((self.source / "geninfo-args.json").read_text())
        directories = [
            Path("build/CMakeFiles/vortex_cxx_shared.dir"),
            Path("build/tests/CMakeFiles/vortex_cxx_test.dir"),
        ]
        self.assertEqual([Path(arg) for arg in args[:2]], directories)
        for directory in directories:
            self.assertTrue(list((self.source / directory).rglob("*.gcda")), f"No coverage data in {directory}")

    def test_inherited_launcher_is_disabled(self) -> None:
        self.env["CMAKE_CXX_COMPILER_LAUNCHER"] = self.launcher
        self.run_report()

    def test_cached_launcher_is_disabled(self) -> None:
        build = self.source / "build"
        # Detect the real compiler before installing the rejecting launcher in the cache.
        self.cmake_configure(self.source, build)
        self.cmake_configure(self.source, build, f"-DCMAKE_CXX_COMPILER_LAUNCHER={self.launcher}")
        self.run_report()


if __name__ == "__main__":
    unittest.main()
