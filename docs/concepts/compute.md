# Vortex Compute

Encoding vtables can define optional implementations of compute functions where it's possible to optimize the
implementation beyond the default behavior of canonicalizing the array and then performing the operation.

For example, `DictEncoding` defines an implementation of compare where given a constant right-hand side argument,
the operation is performed only over the dictionary values and the result is wrapped up with the original dictionary
codes.

## Compute Functions

* `binary_boolean(lhs: ArrayData, rhs: ArrayData, BinaryOperator) -> ArrayData`
    * Compute `And`, `AndKleene`, `Or`, `OrKleene` operations over two boolean arrays.
* `binary_numeric(lhs: ArrayData, rhs: ArrayData, BinaryOperator) -> ArrayData`
    * Compute `Add`, `Sub`, `RSub`, `Mul`, `Div`, `RDiv` operations over two numeric arrays.
* `compare(lhs: ArrayData, rhs: ArrayData, CompareOperator) -> ArrayData`
    * Compute `Eq`, `NotEq`, `Gt`, `Gte`, `Lt`, `Lte` operations over two arrays.
* `try_cast(ArrayData, DType) -> ArrayData`
    * Try to cast the array to the specified data type.
* `fill_forward(ArrayData) -> ArrayData`
    * Fill forward null values with the most recent non-null value.
* `fill_null(ArrayData, Scalar) -> ArrayData`
    * Fill null values with the specified scalar value.
* `invert_fn(ArrayData) -> ArrayData`
    * Invert the boolean values of the array.
* `like(ArrayData, pattern: ArrayData) -> ArrayData`
    * Perform a `LIKE` operation over two arrays.
* `scalar_at(ArrayData, index) -> Scalar`
    * Get the scalar value at the specified index.
* `search_sorted(ArrayData, Scalar) -> SearchResult`
    * Search for the specified scalar value in the sorted array.
* `slice(ArrayData, start, end) -> ArrayData`
    * Slice the array from the start to the end index.
* `take(ArrayData, indices: ArrayData) -> ArrayData`
    * Take the specified nullable indices from the array.
* `filter(ArrayData, mask: Mask) -> ArrayData`
    * Filter the array based on the given mask.