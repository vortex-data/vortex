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

For example:

```c
// DType is an opaque pointer to the Rust DType enum.
// typedef void* DType;


// Pointer to memory allocated by Rust (via `Box::new()`)
DType d_i32 = DType_new(DTYPE_PRIMITIVE_I32, false);

printf("nullable = %d\n", DType_nullable(d_i32));
// "nullable = 1"

// Rust memory is freed, d_i32 is now a dangling pointer.
DType_free(d_i32);
```

## C Strings

C strings are null-terminated, while Rust's are not. This means that unfortunately, we cannot simply return a pointer
to a `&str` or `&String` but instead need to copy the data into a new allocation. Instead, methods that return a string
should instead receive two arguments:
a `*mut c_void` which is a pointer to the start of a buffer that is large enough to hold the largest string, and a
`*mut c_int` to store the length of the buffer after writing.

This means that we can actually request a pointer instead

Because C and Rust have different string representations, functions that return Strings must instead receive
a pointer to a buffer, and a pointer to an integer. Any `str` or `String` from Rust will be copied into the output
buffer,

## Updating

To rebuild the header run

```shell
cbindgen --config cbindgen.toml --crate vortex-ffi --output cinclude/vortex.h
```