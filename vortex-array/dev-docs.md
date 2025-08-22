# High-level overview of `Array`s in `vortex-array`

## `Array`

An `Array` is a type-erased, in-memory representation of a (possibly compressed) logical array of data with known length. Rust types (usually a normal Rust `struct`) implement `Array` plus several other required traits.

`Array`s are recursively compressed, which means that we can apply compression multiple times to data. This also means that `Array`s can be recursively **de**compressed, which allows for partial decompression to reduce the amount of computation necessary.

All data is stored in either a `Buffer<T>` (which is a wrapper around an aligned buffer that takes up space in `O(n)`) or an associated `Metadata` (which takes up space in `O(1)`).

## `DType`

`DType` is an enumeration of the different **logical** types that can be represented in a Vortex array. This is different from physical types, which we call **encodings**, that represent the actual layout of data (compressed or uncompressed). Each logical type has at least one physical array encoding that lives in the `vortex-array` crate (or more formally, the set of physical types is surjective into the set of logical types).

Note that a `DType` represents the logical type of the elements in the `Array`s, **not** the logical type of the `Array` itself.

## Encodings

We can think of an encoding as a “class” (in the OOP sense). `Array`s are simply instances of an encoding. And similar to a class, each `Array` implementation also comes with a "virtual" function table `VTable`, which we will explain later.

### Canonical Encodings

For every unique `DType`, there exists a **canonical encoding** of that logical type. We can think of these canonical encodings as the "default" physical type / encoding of a given logical type. 

All `Array`s can be decompressed into canonical encodings based on the logical `DType`. However, note that canonical encodings **do not** have to be fully decompressed, where we consider "fully decompressed" to mean that the `Array` is zero-copyable to Apache Arrow.

For example, a validity child `Array` (the null map) does not have to be fully decompressed in a canonical encoding (_note that the Vortex maintainers are considering requiring it fully decompressed_). Similarly, the `struct` canonical encoding does not require its field `Array`s (children) to be fully decompressed (canonicalized). In other words, we only care about canonical access to the **components** of the `Array`.

All of the canonical encodings are located in the `vortex-array/src/arrays` module. There are a few other **non-canonical encodings** that live in there as well for convenience.

### Non-Canonical Encodings

Non-canonical encodings are simply physical data layouts / encodings that are _not_ the default representation of a logical `DType`.

There are a few non-canonical encodings in the `vortex-array/src/arrays` module (notably `constant`, `datetime`, `chunked`). The remaining non-canonical encodings live in the `encodings/` subcrates. These non-canonical encodings represent faster or more specialized encodings / compression strategies like dictionary and run-end encoding, as well as some more state-of-the-art compression schemes like `ALP` for floating-point numbers and `FSST` for strings.

The non-canonical (faster / advanced) encodings in the `encodings` directory will generally form a tree structure of `Array`s (some of which may have only one root node), where the leaf nodes of this tree are canonical encodings. There are a few exceptions to this (`fastlanes` has `bitpacking`).

All encodings of arrays implement `VTable`, and all have several **compute kernels** for specialized computation (which we will also talk about later).

## `VTable`

Every `Array` implementation must implement `VTable`, which is conceptually similar to virtual tables used for dynamic dispatch in Rust or C++.

Implementations of `Array` (via help from a zero-sized marker type generated with macros) implement the `VTable` trait, which is a trait that combines several other traits for organization. Some of the behavior includes canonicalizing the array (`CanonicalVTable`) or serializing/deserializing the array (`SerdeVTable`).

### Decompression

Decompression happens via calls to the `canonicalize` method, which `Array`s implement via the trait `CanonicalVTable`. Recall that `Array` are _recursively compressed_ data: we can think of the `canonicalize` method as decompression of the topmost "compression layer" of an `Array` into a canonical encoding. Most of the canonical encodings have relatively straightforward decompression implementations. Non-canonical encodings utilize specialized compute kernels to speed up decompression (which we talk about in the next section).

In order to fully decompress an `Array`, we use the `Array::append_to_builder` method. It takes a `&mut dyn ArrayBuilder` of the same `DType`, which further allows the concatenation of multiple arrays.

## Compute Kernels

All encodings implement several `Kernel`s (all of which are traits from `vortex_array::compute`) such as `TakeKernel`, `FilterKernel`, or `FillNullKernel` which return new `Array`s, or `IsSortedKernel` and `MinMaxKernel` which return other things. These kernels are registered with the `inventory` crate via `register_kernel!`.

Generally speaking, any operation over an array of data that takes work in `O(n)` could be a candidate compute kernel. The kernels that turn naive `O(n)` operations into `O(1)` are generally the most useful in speeding up operations.

As an example for how these compute kernels can speed up decompression, suppose we have a `ConstantArray` of `4` and we want to add `2` to each value in the array (so that the resultant array is a array of just `6`s). Instead of decompressing the `ConstantArray` into a full array and adding the number to each element (`O(n)`), we can instead just add `2` to the constant `4` and return a `ConstantArray` of `6`.

## Walkthrough: Dictionary Decompression

We will step through the code of the canonicalization (decompression) of a simple `DictArray` (which is located in `encodings/dict/src/array.rs` under `impl CanonicalVTable<DictVTable>`).

A `DictArray` has 2 children, `codes` and `values`. We call the `take` function (located in `vortex-array/src/compute/take.rs` to create a new `Array` that is built from the values in `values`, indexed by the `codes`.

Inside `take`, we `invoke` a static `TAKE_FN: ComputeFn`, which holds registered `TakeKernelRef` kernels that have been registered by many of the existing `Array` types via `impl TakeKernel for SomethingVTable`.

`invoke` will call `take_impl`, which will attempt to apply any of the registered compute kernels in order to avoid having to canonicalize the child `values` array. Regardless of what happens, it will return an `Array`, which we then update the statistics for. Finally, we return the back to `take`, and then back to the `canonicalize` method for `DictVTable`, and we are done.
