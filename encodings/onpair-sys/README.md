# vortex-onpair-sys

Low-level FFI bindings to the [OnPair][onpair] short-string compression library.

OnPair is a dictionary-based compressor with **random access** and
**compressed-domain predicate evaluation** (substring, prefix, exact-match),
making it a natural fit for column scans with filter pushdown.

This crate is the unsafe `*-sys` layer used by [`vortex-onpair`][onpair-rs].
End users should depend on `vortex-onpair`, not this crate.

## Build

The build script uses CMake's `FetchContent` to pull
`gargiulofrancesco/onpair_cpp` at the pin recorded in `cmake/onpair_pin.cmake`,
applies a small patch that replaces `boost::unordered_flat_map` with
`std::unordered_map` to avoid the Boost dependency, and compiles both OnPair
and a thin C ABI shim (`cxx/onpair_shim.{h,cpp}`) into a single static archive
that is linked into the Rust crate.

### Requirements

- CMake >= 3.21
- A C++20-capable compiler (GCC >= 11, Clang >= 13, MSVC >= 19.29)
- Network access on first build (for `FetchContent`)

After the first build the source tree is cached under
`$OUT_DIR/onpair-build/_deps`, so subsequent builds are offline.

[onpair]: https://arxiv.org/abs/2508.02280
[onpair-rs]: ../onpair
