# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Real C/C++ Cargo dependencies built through the production CMake entry point."""

import hashlib
import json
import shlex
import shutil
import subprocess
import tomllib
import unittest
from pathlib import Path

from support import CMakeTest, rust_toolchain_environment


def snapshot(*paths: Path) -> dict[Path, tuple[int, bytes]]:
    return {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest()) for path in paths}


class CompilerCommandTests(CMakeTest):
    def setUp(self) -> None:
        super().setUp()
        packages = tomllib.loads((self.repo / "Cargo.lock").read_text())["package"]
        [cc] = [package for package in packages if package["name"] == "cc"]

        self.source = self.work / "source"
        shutil.copytree(Path(__file__).parent / "fixtures/native", self.source)
        include = self.source / "native-helper/include directory's"
        self.write(include / "fixture.h", "#define HEADER_VALUE 3\n")
        ffi = self.source / "vortex-ffi"
        production = self.repo / "vortex-ffi"
        shutil.copytree(production / "cmake", ffi / "cmake", ignore=shutil.ignore_patterns("tests"))
        shutil.copyfile(production / "CMakeLists.txt", ffi / "CMakeLists.txt")

        manifest = self.source / "native-helper/Cargo.toml.in"
        self.write(manifest.with_suffix(""), manifest.read_text().replace("@CC_VERSION@", cc["version"]))
        self.env = rust_toolchain_environment()
        for key in list(self.env):
            if key.startswith(("CC", "CXX", "CFLAGS", "HOST_", "TARGET_", "SCCACHE_", "CMAKE_")) or key in (
                "AR",
                "RANLIB",
                "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_TARGET_DIR",
            ):
                self.env.pop(key)
        self.env.update(CARGO_NET_OFFLINE="true", CARGO_BUILD_JOBS="2")
        self.command("cargo", "generate-lockfile", "--offline", cwd=self.source)

    def configure(
        self,
        value=7,
        argument=7,
        policy=True,
        instrumentation=False,
        generator="Ninja",
        build_name=None,
        debug_info=None,
    ) -> None:
        self.build_dir = self.work / (build_name or f"{generator} build directory's")
        self.target_dir = self.build_dir / "ffi/cargo-target"
        options = []
        if debug_info is not None:
            options.append(f"-DVORTEX_DEBUG_INFO={debug_info}")
        include = self.source / "native-helper/include directory's"
        errors = ["-Werror", "-Werror=unused-variable", "-pedantic-errors"] if policy else []
        for language, compiler in (("C", "clang"), ("CXX", "clang++")):
            mandatory = [f"-DREQUIRED_{language}={argument}", f"-I{include}"]
            flags = ["-Wunused-variable", f"-DPARENT_VALUE={value}", *errors]
            if instrumentation:
                mandatory.append("--coverage")
                flags.append("-fsanitize=undefined")
            options += [
                f"-DCMAKE_{language}_COMPILER={shutil.which(compiler)}",
                f"-DFIXTURE_{language}_ARG1={shlex.join(mandatory)}",
                f"-DCMAKE_{language}_FLAGS={shlex.join(flags)}",
                f"-DCMAKE_{language}_FLAGS_DEBUG={shlex.join(['-O1', '-DCONFIG_VALUE=1', *errors])}",
            ]
        self.cmake_configure(self.source, self.build_dir, "-DCMAKE_BUILD_TYPE=Debug", *options, generator=generator)

    def build(self, target: str = "vortex_ffi_cargo_build", success: bool = True) -> subprocess.CompletedProcess[str]:
        return self.cmake_build(self.build_dir, "--target", target, success=success)

    def archives(self) -> dict[Path, tuple[int, bytes]]:
        archives = sorted(self.target_dir.rglob("libnative_*.a"))
        self.assertEqual(len(archives), 4, archives)
        return snapshot(*archives)

    def test_compiler_arguments_warning_policy_and_freshness(self) -> None:
        self.configure()
        rejected = self.build("parent_native", success=False)
        self.assertIn("error: unused variable 'vendored_unused'", rejected.stdout + rejected.stderr)
        self.build()
        original = self.archives()
        for archive in original:
            self.assertIn("warning: unused variable 'vendored_unused'", (archive.parent.parent / "output").read_text())
        self.build()
        self.assertEqual(self.archives(), original, "An unchanged build must remain fresh")
        self.configure(policy=False)
        self.build("parent_native")
        self.build()
        self.assertEqual(self.archives(), original, "Parent warning-as-error policy must not invalidate Cargo")
        for changes in ({"value": 8}, {"value": 8, "argument": 9}):
            with self.subTest(changes=changes):
                self.configure(policy=False, **changes)
                self.build()
                changed = self.archives()
                self.assertEqual(changed.keys(), original.keys())
                for path in original:
                    self.assertNotEqual(changed[path][0], original[path][0], path)
                    self.assertNotEqual(changed[path][1], original[path][1], path)
                original = changed

    def test_header_lifecycle(self) -> None:
        # Incremental rustc changes archive member names even when the object bytes are identical.
        self.env["CARGO_INCREMENTAL"] = "0"
        for generator in ("Ninja", "Unix Makefiles"):
            with self.subTest(generator=generator):
                version = self.write(self.source / "vortex-ffi/header-version", "1\n")
                # Ninja < 1.13 splits apostrophes in compiler depfiles. Other tests retain
                # apostrophe-containing build paths to cover the Cargo compiler handoff.
                self.configure(generator=generator, build_name=f"{generator} build directory")
                consumer = self.build_dir / "header_consumer"
                source_header = self.source / "vortex-ffi/cinclude/vortex.h"
                staged = self.build_dir / "ffi/vortex-artifacts/include/vortex.h"
                archive = self.build_dir / "ffi/vortex-artifacts/libvortex_ffi.a"
                self.build("header_consumer")
                self.assertEqual(self.command(consumer).stdout.strip(), "1")
                archive_state = snapshot(archive)
                self.write(version, "2\n")
                self.build("header_consumer")
                self.assertEqual(self.command(consumer).stdout.strip(), "2", "Must update on the first build")
                self.assertEqual(snapshot(archive), archive_state, "Header-only change must leave archive identical")
                self.assertEqual(staged.read_bytes(), source_header.read_bytes())
                outputs = snapshot(archive, staged, source_header, consumer, *self.build_dir.rglob("*.o"))
                self.build("header_consumer")
                self.assertEqual(snapshot(*outputs), outputs)
                self.build("clean")
                for path in (staged.parent, archive, self.target_dir):
                    self.assertFalse(path.exists(), path)
                self.assertEqual(snapshot(source_header), {source_header: outputs[source_header]})
                self.build("header_consumer")
                self.assertEqual(self.command(consumer).stdout.strip(), "2")
                self.assertEqual(staged.read_bytes(), source_header.read_bytes())

    def test_host_target_instrumentation_and_cache_boundary(self) -> None:
        log = self.work / "cache calls.jsonl"
        # Record the cache boundary, not cache behavior; every call still runs the real compiler.
        self.env["RUSTC_WRAPPER"] = self.executable(
            "cache tools' directory/sccache",
            f"""\
            import json, os, sys
            if '-c' in sys.argv and any(arg.endswith(('native.c', 'native.cpp')) for arg in sys.argv):
                with open({str(log)!r}, 'a') as log:
                    log.write(json.dumps([bool(os.environ.get('CARGO_ENCODED_RUSTFLAGS')), sys.argv[1:]]) + '\\n')
            os.execv(sys.argv[1], sys.argv[1:])
            """,
        )
        self.configure(instrumentation=True, debug_info=0)
        self.build()
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sorted(target for target, _ in calls), [False, False, True, True])
        for target, args in calls:
            self.assertIn(Path(args[0]).name, ("clang", "clang++"))
            self.assertEqual([arg for arg in args if arg.startswith("-g")][-1:], ["-g0"], args)
            for flag in ("--coverage", "-fsanitize=undefined"):
                self.assertEqual(flag in args, target, args)
        objects = sorted(self.target_dir.rglob("out/*-native.o"))
        self.assertEqual(len(objects), 4, objects)
        for obj in objects:
            with self.subTest(object=obj):
                target = obj.relative_to(self.target_dir).parts[0] != "debug"
                symbols = self.command("nm", "-u", obj).stdout
                for runtime in ("ubsan", "llvm_gcda"):
                    self.assertEqual(runtime in symbols, target, symbols)
                self.assertEqual(obj.with_suffix(".gcno").exists(), target)


if __name__ == "__main__":
    unittest.main()
