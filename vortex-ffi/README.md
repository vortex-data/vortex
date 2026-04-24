# Foreign Function Interface

Vortex is a file format that can be used by any execution engine. Nearly every programming language supports
the C ABI (Application Binary Interface), so by providing an FFI interface to work with Vortex objects we can
make it easy to support a variety of languages.

Check out the [`examples`](./examples/) directory to see an example of how to use the API to build
a real native application.

## Design

The FFI is designed to be very simple and follows a very object-oriented approach:

- **Constructors** are simple C functions that return opaque pointers
- **Methods** are functions that receive an opaque pointer as the first argument, followed by subsequent arguments.
  Methods may return a value or void.
- **Destructors** free native resources (allocations, file handles, network sockets) and must be explicitly called by
  the foreign language to avoid leaking resources.

Constructors will generally allocate rust memory, and destructors free that memory.

## Documentation

The FFI API is documented in `docs/api/c` with explicit inclusion of types, enums, and functions, etc. Note that an
item cannot be referenced in the documentation if it does not have a documentation comment.

## Updating Headers

To rebuild the header file:

```sh
cargo +nightly build -p vortex-ffi
```

The header generation uses cbindgen's macro expansion feature which requires nightly.
Stable builds use the checked-in header file at `cinclude/vortex.h`.

### Testing C part

Build the test library

```sh
cmake -Bbuild
cmake --build build -j $(nproc)
```

Run the tests

```sh
ctest --test-dir build -j $(nproc)
```

You would need C++ compiler toolchain to run the tests since they use Catch2.

### Testing Rust part with sanitizers

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

- `-Zbuild-std` is needed as memory and thread sanitizers report std errors
  otherwise.
- `--no-default-features` is needed as we use Mimalloc otherwise which interferes
with sanitizers.
- `allow-abi-mismatch` is safe because in our dependency graph only crates like
  `compiler_builtins` unset sanitization, and they do it on purpose.
- Make sure to use `cargo test` and not `cargo nextest` as nextest reports less
leaks.
- If you want stack trace symbolization, install `llvm-symbolizer`.

### Testing Rust and C with sanitizers

1. Build FFI library with external sanitizer runtime:

```sh
RUSTFLAGS="-Zsanitizer=address -Zexternal-clangrt" \
cargo +nightly build -Zbuild-std --target=<target triple> \
--no-default-features -p vortex-ffi
```

2. Build tests with target triple

```sh
cmake -Bbuild -DWITH_ASAN=1 -DTARGET_TRIPLE=<target triple>
```

3. Run the tests (ctest doesn't output failures in detail):

```sh
./build/test/vortex_ffi_test 2>& 1 | rustfilt -i-
```
