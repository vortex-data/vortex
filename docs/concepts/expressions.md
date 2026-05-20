# Expressions

Expressions in Vortex are used to describe scalar operations over arrays. For example, when scanning a Vortex file a
user will pass both a filter expression that must resolve to a boolean, and a projection expression that can be applied
to the returned data.

## Scalar Functions

Expressions are defined as abstract scalar functions. These vtables define the signature of the function, properties
such as whether the function is null-sensitive, and the actual logic for executing the function over input arrays.

The built-in scalar functions can be found in the `vortex-array::expr` module, with additional use-case specific
functions provided by integration and plugin crates.

In the future, we plan to add full support for geospatial functions, date-time functions, and more.

## Scalar Function Arrays

Since Vortex aims to defer as much computation as possible, applying an expression to an array doesn't immediately
compute the result. Instead, the computation is deferred by constructing a new array that represents the application
of the expression to the input array(s). This new array is always a [`ScalarFnArray`] and can be used by other arrays
to perform push-down computation or define specialized execution kernels.

For example, if a Vortex file contains a single array:

```
bitpacked:
  bitwidth: 4
  buffer: Buffer<u32>
```

And the projection expression `{x: $, y: $ + 1}` is applied to it (where `$` represents the expression's scope), then
the resulting array before simplification will look a little bit like this:

```
scalar_fn(struct.pack):
  names: ["x", "y"]
  inputs:
    - bitpacked:
        bitwidth: 4
        buffer: Buffer<u32>
    - scalar_fn(binary.add):
        inputs:
          - bitpacked:
              bitwidth: 4
              buffer: Buffer<u32>
          - constant(1)
```

Delaying the computation in this way allows for many more optimizations to be performed later, including fused 
computation of several expressions at once, as is commonly done by the Vortex GPU compute context.

## Type Checking

Vortex expressions are strictly typed. This means that the input data types to an expression must exactly match the
expected data types of the expression's signature.

For example, all binary functions require the same data type for both inputs. Any casts should be performed by the caller
before constructing the expression. This allows Vortex to be agnostic to the type conversion rules of different compute
engines.

The notable exception to this rule is nullability. For example, the equality comparison function allows comparing a
`u32` and `u32?` array, but not a `u32` and `i32` array.
