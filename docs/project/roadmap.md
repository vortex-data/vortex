# Roadmap

This page outlines the major projects that are actively in development and those planned for the
near future.

## In-Flight Projects

### CUDA GPU Support

Adding support for GPU-accelerated compute over Vortex arrays using CUDA. This will allow
encodings to be decompressed and evaluated directly on the GPU.

### Lazy Array Evaluation

Reworking array compute to use lazy evaluation, where operations build up an expression graph that
is materialized on demand. This is a prerequisite for efficient GPU support, since it allows the
runtime to batch and schedule work across devices.

### Extension DTypes

Introducing user-defined logical types that extend the built-in Vortex type system. Extension
DTypes allow libraries and applications to register custom types with their own semantics while
still benefiting from Vortex's encoding, compression, and I/O infrastructure.

## Upcoming Projects

### Scan API

An abstract table-scan interface that positions Vortex as an interchange layer between data sources
and query engines. The Scan API will support pluggable data sources and can be consumed over the
C ABI for in-process integrations or over RPC for remote/distributed access.

### Language Bindings Overhaul

A comprehensive rework of the language bindings to expose plugin and extension points across the
ecosystem:

- **Rust, C, C++, and Python** will have first-class support for extending Vortex with custom
  encodings, compute functions, and DTypes.
- **Other languages** (e.g. Java) will initially focus on reading and writing Arrow data to and
  from Vortex files.

### Tensor Extension DType

A built-in extension DType for multi-dimensional tensor data, enabling native support for
fixed-shape and variable-shape tensors within Vortex arrays and files.

### Variant DType

A DType for representing arbitrarily nested, JSON-like data within Vortex arrays and files. This
enables efficient columnar storage and querying of semi-structured data.
