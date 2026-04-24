# Embedding Vortex

:::{warning}
This section is under construction. For guidance on embedding Vortex, please join the
[Vortex Slack channel](https://join.slack.com/t/vortex-array/shared_invite/zt-2ycp2w24h-sRdrGbMGPmQwCuPQT40Jig)
or start a [GitHub Discussion](https://github.com/spiraldb/vortex/discussions).
:::

Vortex can be embedded into applications and services via its C FFI, C++ wrapper, or the Scan API.
The following topics are planned for this section:

- **C FFI** -- the Vortex C API, building and linking, session management, arrays, dtypes,
  error handling, and memory ownership.
- **C++** -- the C++ wrapper around the C FFI, CMake integration, and RAII wrappers.
- **Scan API** -- serving Vortex data to query engines, wire format serialization, filter and
  projection pushdown, and custom scan providers.
- **GPU Acceleration** -- CUDA requirements, GPU-accelerated decompression and compute,
  host/device memory management, and current limitations.

```{toctree}
---
hidden: true
---

ffi
cxx
scan-api
gpu
```
