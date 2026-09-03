# vortex-velox

`vortex-velox` defines the native adapter contract between Vortex and Velox.
The crate keeps Vortex API use inside the Vortex repository. Velox implements
the engine side of the versioned C ABI.

The crate is private and ships as part of a pinned Vortex source release. It is
not a general Vortex C API.

Native primitive visits copy values and validity into compact owned buffers.
The value allocation uses `uint64_t` alignment. The ABI reports the exact
allocation size that the owner retains. This contract avoids an understated
charge for a slice of a larger Vortex allocation.

Before Arrow fallback conversion, the host reserves a conservative byte count.
The adapter stops before Arrow allocations if the host rejects the reservation.
After conversion, the adapter refunds the difference from retained payload
capacities. If actual capacity exceeds the reservation, the adapter requests
the difference before it returns outputs. Arrow release frees the final charge.

## Contract boundary

The adapter exposes a versioned C ABI in `cinclude/vortex_velox.h`. The header
and static library are standalone. Velox calls only `vx_velox_*` symbols.
Opaque handle layouts stay inside Vortex.

The adapter accepts host callbacks for random reads. Velox can implement those
callbacks with `dwio::common::BufferedInput`, so Vortex uses the existing cache
and file-system path. Vortex checks the host cancellation callback before each
read callback. This check does not interrupt an active callback or CPU work.

Vortex can call read callbacks concurrently. The host callback context must be
thread-safe. Each thread owns its callback error string until its next callback.
Callbacks must catch exceptions and must not unwind across the C ABI.

Each source reports natural row ranges. Velox maps byte ownership to these row
ranges. The adapter maps projection and filter expressions to Vortex scan
requests. Metadata exclusion returns only splits that Vortex proves cannot
contain a match.

Source schemas cross the boundary through the Arrow C Data Interface. The
adapter does not maintain a second recursive type protocol.

Native array visits cover primitive arrays and structural wrappers. The Arrow
C Data fallback covers arrays without a native visit. Both paths transfer
ownership through explicit release callbacks.

## Build

Build the static adapter library from the workspace root:

```bash
cargo build --locked --package vortex-velox
```

The Vortex `rust-toolchain.toml` file selects the Rust toolchain. Velox can use
a pinned Vortex archive or a local checkout.

Run the adapter tests with:

```bash
cargo test --locked --package vortex-velox
```

The package tests compile the C and C++ header contracts. Production builds do
not compile or link contract helper symbols.

The crate does not construct Velox vectors. Velox owns vector construction,
lazy-load policy, value hooks, mutation semantics, and memory accounting.
