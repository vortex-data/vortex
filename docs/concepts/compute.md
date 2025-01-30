# Vortex Compute

Encoding vtables can define optional implementations of compute functions where it's possible to optimize the
implementation beyond the default behavior of canonicalizing the array and then performing the operation.

For example, `DictEncoding` defines an implementation of compare where given a constant right-hand side argument,
the operation is performed only over the dictionary values and the result is wrapped up with the original dictionary
codes.

## Compute Functions

* `binary_boolean(lhs: Array, rhs: Array, BinaryOperator) -> Array`
    * Compute `And`, `AndKleene`, `Or`, `OrKleene` operations over two boolean arrays.
* `binary_numeric(lhs: Array, rhs: Array, BinaryOperator) -> Array`
    * Compute `Add`, `Sub`, `RSub`, `Mul`, `Div`, `RDiv` operations over two numeric arrays.
* `compare(lhs: Array, rhs: Array, CompareOperator) -> Array`
    * Compute `Eq`, `NotEq`, `Gt`, `Gte`, `Lt`, `Lte` operations over two arrays.
* `try_cast(Array, DType) -> Array`
    * Try to cast the array to the specified data type.
* `fill_forward(Array) -> Array`
    * Fill forward null values with the most recent non-null value.
* `fill_null(Array, Scalar) -> Array`
    * Fill null values with the specified scalar value.
* `invert_fn(Array) -> Array`
    * Invert the boolean values of the array.
* `like(Array, pattern: Array) -> Array`
    * Perform a `LIKE` operation over two arrays.
* `scalar_at(Array, index) -> Scalar`
    * Get the scalar value at the specified index.
* `search_sorted(Array, Scalar) -> SearchResult`
    * Search for the specified scalar value in the sorted array.
* `slice(Array, start, end) -> Array`
    * Slice the array from the start to the end index.
* `take(Array, indices: Array) -> Array`
    * Take the specified nullable indices from the array.
* `filter(Array, mask: Mask) -> Array`
    * Filter the array based on the given mask.