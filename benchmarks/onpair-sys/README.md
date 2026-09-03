# vortex-onpair-sys

Low-level FFI bindings to the [OnPair][onpair] short-string compression library.

OnPair is a dictionary-based compressor with **random access** and
**compressed-domain predicate evaluation** (substring, prefix, exact-match),
making it a natural fit for column scans with filter pushdown.

This is a standalone, benchmark-only adapter for comparing Vortex's Rust
implementation with the original C++/Boost implementation. It is excluded from
the Vortex workspace and is never compiled by production or workspace builds.

## Build

The build script uses CMake's `FetchContent` to pull
`gargiulofrancesco/onpair_cpp` at the pin recorded in `cmake/onpair_pin.cmake`,
uses its native `boost::unordered_flat_map` implementation, and compiles both
OnPair and a thin C ABI shim (`cxx/onpair_shim.{h,cpp}`) into a single static
archive that is linked into the Rust crate. If the host has no compatible
Boost.Unordered installation, the pinned upstream CMake build fetches one.

### Requirements

- CMake >= 3.21
- A C++20-capable compiler (GCC >= 11, Clang >= 13, MSVC >= 19.29)
- Boost.Unordered >= 1.81, fetched automatically when unavailable
- Network access on first build (for `FetchContent`)

After the first build the source tree is cached under
`$OUT_DIR/onpair-build/_deps`, so subsequent builds are offline.

[onpair]: https://arxiv.org/abs/2508.02280
