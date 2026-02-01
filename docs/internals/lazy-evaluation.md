# Work-in-progress: Lazy Evaluation in Vortex

This guide intends to provide an overview of the in-flight and upcoming changes to Vortex to enable
fully lazy evaluation of Vortex arrays.

Hopefully this document helps users and contributors understand the design decisions and plan around
the upcoming breaking API changes required to implement this feature.

The motivation for this work comes in many parts, including:

* Support for alternate execution models such as GPU, pipelined CPU, or JIT-compiled CPU.
* Improved scan performance with common-subtree elimination.
* Improved visibility into the optimizations that Vortex applies by making the computation graph explicit.
* Easier to benchmark and improvement performance of individual compute functions by isolating them from
  lazy decompression logic.
* Easier to extend Vortex with new compute functions, such as geo-spatial functionality.
* Simpler to implement custom arrays and layouts by reducing the API surface area.
* Enabling more advanced statistics and pruning such as using bloom filters and free-text indexes.

## Summary of Changes

* Define `vortex-vector` as a fully decompressed in-memory format used for CPU computation.
* Vortex `Array` to represent a logical decompression plan.
* Introduce `ScalarFn` to define semantics and implementation of scalar compute over Vortex vectors.
* Make `Expression` a non-pluggable closed enum. Plugins will implement `ScalarFn` instead.
    * Note this avoids the current situation we're in where all arrays need to know about all compute functions.
* Introduce `ScalarFnArray` to represent lazy application of a `ScalarFn` over one or more Vortex arrays.
    * Existing compute function dispatch is re-implemented as Array optimization rules.
* Redesign the `Layout` API to use simpler optimization rules instead of complex expression partitioning.
* Implement statistics falsification as optimizer rules over expressions.
    * e.g. `falsify(a > 10)` becomes `stat.max(a) <= 10`.
    * This also enables custom falsification expressions such as bloom filter checks.
