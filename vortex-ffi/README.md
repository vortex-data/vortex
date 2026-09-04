# Vortex C bindings

## Runtime threading

The FFI uses a shared, caller-driven runtime. By default Vortex creates no runtime worker threads:
the host threads currently executing FFI calls drive the runtime. Multiple host threads making
concurrent FFI calls can drive runtime work in parallel while keeping thread ownership entirely in
the host application.

Applications that want a single FFI operation to make progress on additional threads may opt into
Vortex-owned background workers with `vx_runtime_set_worker_threads`. This is a process-global
setting shared by every FFI session. Calling it with a non-zero count changes the threading model
from host-thread-only execution to a combination of host threads and Vortex-owned workers. Calling
it with zero signals the background workers to stop and restores the host-thread-only
configuration.

Applications that already supply concurrency through their own host threads should leave the
worker count at its default of zero to avoid oversubscription.

## Updating Headers

If you're developing FFI and want to rebuild `cinclude/vortex.h`, run:

```sh
cargo +nightly build -p vortex-ffi
```

## Usage from a CMake project

CMake builds the Rust archive through Cargo; no separate `cargo build` is needed. Add the
repository root, or this directory alone, and link the static target:

```cmake
add_subdirectory(path/to/vortex vortex)
target_link_libraries(my_target PRIVATE Vortex::ffi_static)
```

The target carries the headers in `cinclude/`, the archive, and the system libraries it needs.
Build options such as `VORTEX_CARGO_PROFILE`, `VORTEX_ENABLE_CUDA`, and `VORTEX_SANITIZER` are
documented in the [C++ README](../lang/cpp/README.md) and apply to both layers.

## Running C examples

```sh
cmake -S . -B build -DVORTEX_BUILD_EXAMPLES=ON
cmake --build build --parallel
./build/examples/write_sample sample.vortex
./build/examples/dtype 'sample.vortex'
./build/examples/scan 'sample.vortex'
./build/examples/scan_to_arrow 'sample.vortex'
```

## Testing C part

The tests use Catch2, so a C++ compiler is required:

```sh
cmake -S . -B build -DVORTEX_BUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Testing Rust part with sanitizers

The Rust tests run under a sanitizer with nightly `cargo test`. Substitute the native target
triple, for example `x86_64-unknown-linux-gnu`:

```sh
# inside vortex-ffi
RUSTFLAGS="-Zsanitizer=address -Cunsafe-allow-abi-mismatch=sanitizer" \
cargo +nightly test -Zbuild-std --target <target triple> --tests -- --no-capture
```

Use `-Zsanitizer=memory` for MemorySanitizer and `-Zsanitizer=thread` for ThreadSanitizer; the
latter needs `TSAN_OPTIONS="suppressions=$PWD/tsan_suppressions.txt"`.

- `-Zbuild-std` is needed as memory and thread sanitizers report std errors otherwise.
- `allow-abi-mismatch` is safe because in our dependency graph only crates like `compiler_builtins`
  unset sanitization, and they do it on purpose.
- `--tests` skips doctests, which rustdoc builds without `RUSTFLAGS` and which would therefore
  mismatch the sanitizer-built dependencies.
- Make sure to use `cargo test` and not `cargo nextest` as nextest reports less leaks.
- If you want stack trace symbolization, install `llvm-symbolizer`.

## Testing Rust and C with sanitizers

CMake instruments the Rust archive, its C dependencies, and the tests together. `VORTEX_SANITIZER`
takes a comma-separated list of `asan`, `lsan`, `ubsan`, and `tsan`. Each instruments the C and C++
code, and all but `ubsan` also instrument the Rust code, since rustc has no UBSan. Rust
instrumentation uses rustup's `nightly` toolchain, which needs the `rust-src` component,
unless `RUSTUP_TOOLCHAIN` selects another, and the C and C++ side needs Clang:

```sh
cmake -S . -B build \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DVORTEX_SANITIZER=asan,ubsan -DVORTEX_SANITIZE_RUST_STD=ON \
    -DVORTEX_BUILD_TESTING=ON
cmake --build build --parallel
./build/test/vortex_ffi_test 2>&1 | rustfilt -i-
```

For ThreadSanitizer use `tsan` and point `TSAN_OPTIONS` at `tsan_suppressions.txt`.
