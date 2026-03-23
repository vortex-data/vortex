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

To rebuild the header file (requires nightly toolchain):

```shell
cargo +nightly build -p vortex-ffi
```

The header generation uses cbindgen's macro expansion feature which requires nightly.
Stable builds use the checked-in header file at `cinclude/vortex.h`.

### Development Workflow

- **For header changes**: Use nightly toolchain to regenerate headers after modifying FFI code
- **For regular development**: Stable toolchain builds work with existing checked-in headers
- **CI validation**: Automated checks verify header freshness using nightly toolchain

### Testing

Build the test library

```
cmake -Bbuild
cmake --build build -j $(nproc)
```

Run the tests

```
ctest --test-dir build -j $(nproc)
```
