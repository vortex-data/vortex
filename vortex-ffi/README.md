# Vortex C bindings

## Use from CMake

Add a complete Vortex checkout. CMake builds the Rust archive for you:

```cmake
add_subdirectory(path/to/vortex vortex)
target_link_libraries(my_target PRIVATE Vortex::ffi_shared)
```

The target supplies the library and headers. For static linkage, use `Vortex::ffi_static`.
Both targets are available regardless of `BUILD_SHARED_LIBS`. You can also add `vortex-ffi`
directly. See the [CMake build guide](../lang/cpp/README.md) for requirements, options, and
deployment limits.

### Examples and tests

Run from the repository root. Tests use Catch2 and require a C++ compiler.

```sh
cmake -S vortex-ffi -B build/ffi \
    -DVORTEX_BUILD_EXAMPLES=ON -DVORTEX_BUILD_TESTS=ON
cmake --build build/ffi --parallel
ctest --test-dir build/ffi --output-on-failure

./build/ffi/examples/write_sample sample.vortex
./build/ffi/examples/dtype sample.vortex
./build/ffi/examples/scan sample.vortex
./build/ffi/examples/scan_to_arrow sample.vortex
```

## Runtime threading

By default, calling threads drive the shared runtime with no Vortex workers.

`vx_runtime_set_worker_threads(n)` configures a library-local pool of background workers that
drive the shared runtime. The pool is shared by all FFI sessions in that library instance and
persists across calls for the lifetime of the library. Passing zero signals the workers to stop
and restores caller-driven execution. Leave it at zero if your application already provides concurrency.

## Update the C header

Regenerate `vortex-ffi/cinclude/vortex.h`:

```sh
cargo +nightly build -p vortex-ffi
```

## Sanitizer tests

### Rust and C/C++ together

Requires nightly Rust and upstream LLVM Clang. See the
[sanitizer guide](../lang/cpp/README.md#sanitizers) for scope and alternatives.

```sh
rustup toolchain install nightly --component rust-src

cmake -S vortex-ffi -B build/ffi-asan \
    -DCMAKE_BUILD_TYPE=Debug -DVORTEX_RUSTUP_TOOLCHAIN=nightly \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DVORTEX_SANITIZER=asan,ubsan -DVORTEX_SANITIZE_RUST_STD=ON \
    -DVORTEX_BUILD_TESTS=ON
cmake --build build/ffi-asan --parallel
ctest --test-dir build/ffi-asan --output-on-failure
```

For ThreadSanitizer, use `-DVORTEX_SANITIZER=tsan` and set
`TSAN_OPTIONS=suppressions=/absolute/path/to/vortex/vortex-ffi/tsan_suppressions.txt`.

Pipe `build/ffi-asan/test/vortex_ffi_test` output through `rustfilt -i-` to demangle Rust symbols.

### Rust only

Use nightly with `rust-src` installed. This example uses the native Linux x86_64 target:

```sh
RUSTFLAGS="-Zsanitizer=address -Cunsafe-allow-abi-mismatch=sanitizer" \
cargo +nightly test -p vortex-ffi -Zbuild-std \
    --target x86_64-unknown-linux-gnu --tests -- --no-capture
```

Substitute your native target. Use `-Zsanitizer=memory` for MemorySanitizer or
`-Zsanitizer=thread` for ThreadSanitizer, with the suppression file above.

- `-Zbuild-std` instruments std to avoid memory/thread sanitizer false positives.
- `-Cunsafe-allow-abi-mismatch=sanitizer` allows deliberately uninstrumented dependencies.
- `--tests` excludes doctests, which do not inherit `RUSTFLAGS`.
- Prefer `cargo test` over nextest for leak detection. Use `llvm-symbolizer` for stack traces.
