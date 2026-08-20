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

```
# in vortex folder
cargo build --release -p vortex-ffi

# in your CMakeLists.txt
include_directory(vortex/vortex-ffi)
target_link_libraries(my_target, vortex_ffi_shared)
# or target_link_libraries(my_target, vortex_ffi)
```

## Running C examples

```sh
cmake -Bbuild -DBUILD_EXAMPLES=1
cmake --build build
./build/examples/dtype
./build/examples/scan
./build/examples/scan_to_arrow
./build/examples/write_sample
```

## Testing C part

Build the test library:

```sh
cmake -Bbuild -DBUILD_TESTS=1
cmake --build build
```

Run the tests:

```sh
ctest --test-dir build -j $(nproc)
```

You will need C++ compiler toolchain to run the tests since they use Catch2.

## Testing Rust part with sanitizers

AddressSanitizer:

```sh
# inside vortex-ffi
RUSTFLAGS="-Z sanitizer=address" \
cargo +nightly test -Zbuild-std \
    --no-default-features --target <target triple> \
    -- --no-capture
```

MemorySanitizer:

```sh
RUSTFLAGS="-Z sanitizer=memory -Cunsafe-allow-abi-mismatch=sanitizer" \
cargo +nightly test -Zbuild-std \
    --no-default-features --target <target triple> \
    -- --no-capture
```

ThreadSanitizer:

```sh
TSAN_OPTIONS="suppressions=$HOME/vortex/vortex-ffi/tsan_suppressions.txt" \
RUSTFLAGS="-Z sanitizer=thread -Cunsafe-allow-abi-mismatch=sanitizer" \
cargo +nightly test -Zbuild-std \
    --no-default-features --target <target triple> \
    -- --no-capture
```

- `-Zbuild-std` is needed as memory and thread sanitizers report std errors otherwise.
- `--no-default-features` is needed as we use Mimalloc otherwise which interferes with sanitizers.
- `allow-abi-mismatch` is safe because in our dependency graph only crates like `compiler_builtins`
  unset sanitization, and they do it on purpose.
- Make sure to use `cargo test` and not `cargo nextest` as nextest reports less leaks.
- If you want stack trace symbolization, install `llvm-symbolizer`.

## Testing Rust and C with sanitizers

1. Build FFI library with external sanitizer runtime:

```sh
RUSTFLAGS="-Zsanitizer=address -Zexternal-clangrt" \
cargo +nightly build -Zbuild-std --target=<target triple> \
    --no-default-features -p vortex-ffi
```

2. Build tests with target triple:

```sh
cmake -Bbuild -DSANITIZER=asan -DTARGET_TRIPLE=<target triple>
```

3. Run the tests (ctest doesn't output failures in detail):

```sh
./build/test/vortex_ffi_test 2>& 1 | rustfilt -i-
```
