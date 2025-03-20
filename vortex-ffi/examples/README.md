# C FFI example

This example shows how to interface with the FFI of this crate using C code.

Run `make` to build the `hello_vortex` binary.

The binary expects a single argument, which is the path to a Vortex file. A new streaming
scan will be created that will materialize file splits as Vortex in-memory arrays, and print
some information about them.

Here's an example from the TPC-H `partsupp` dataset:

```
❯ ./hello_vortex file:///tmp/partsupp.vortex
Scanning file: file:///tmp/partsupp.vortex
Chunk 0: 65536
Chunk 1: 65536
Chunk 2: 65536
Chunk 3: 65536
Chunk 4: 65536
Chunk 5: 65536
Chunk 6: 65536
Chunk 7: 65536
Chunk 8: 65536
Chunk 9: 65536
Chunk 10: 65536
Chunk 11: 65536
Chunk 12: 13568
Scanning complete
```
