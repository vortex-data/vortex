# Vortex C bindings

Building requires CMake 3.25+ and a C11 compiler.

```cmake
add_subdirectory(path/to/vortex vortex)
target_link_libraries(my_target PRIVATE Vortex::ffi_static)
target_link_libraries(my_target PRIVATE Vortex::ffi_shared)
```

Note this builds Vortex Rust side for you.

Tests require a C++-20 compiler. To run tests,

```sh
cmake -S vortex-ffi -B build/ffi -DVORTEX_BUILD_TESTS=ON
cmake --build build/ffi --parallel
ctest --test-dir build/ffi --output-on-failure
```

To run examples,

```sh
cmake -S vortex-ffi -B build/ffi -DVORTEX_BUILD_EXAMPLES=ON
cmake --build build/ffi --parallel

./build/ffi/examples/write_sample sample.vortex
./build/ffi/examples/dtype sample.vortex
./build/ffi/examples/scan sample.vortex
./build/ffi/examples/scan_to_arrow sample.vortex
```

To regenerate `vortex.h` after making some changes, run `cargo +nightly build -p
vortex-ffi`.

## Running tests with sanitizers

C tests with sanitizer also require clang. To run them,

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

To run only Rust-side tests with sanitizers on x86_64,

```sh
# -Zsanitizer=memory and -Zsanitizer=thread are other options
RUSTFLAGS="-Zsanitizer=address -Cunsafe-allow-abi-mismatch=sanitizer" \
cargo +nightly test -p vortex-ffi -Zbuild-std \
    --target x86_64-unknown-linux-gnu --tests -- --no-capture
```

## Driving the runtime

Vortex is an async library which means it requires an underlying runtime, a
thread scheduler. By default, Vortex uses host threads to drive the runtime,
i.e. calling a Vortex function like `vx_data_source_scan` on multiple threads
drives the runtime on the caller threads. You can create an explicit thread pool
inside Vortex by calling `vx_runtime_set_worker_threads(n)`. After this call,
Vortex will drive the runtime on the internal thread pool and not host threads.
