// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the tensor scalar functions add to the row-function machinery: an element type that reads a
//! tensor row and the width rule they share.

use std::marker::PhantomData;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::InputElement;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;
use crate::utils::validate_tensor_float_inputs;

/// The width rule the tensor scalar functions share: every argument is the same float tensor dtype,
/// and the width is its element ptype.
pub(crate) fn tensor_element_ptype(args: &[DType]) -> VortexResult<PType> {
    Ok(validate_tensor_float_inputs(args)?.element_ptype())
}

/// Marker for tensor-valued input elements: accepts any tensor-like extension column whose
/// elements are `T`, and presents each row as its flat elements, `&[T]`.
pub struct TensorRow<T>(PhantomData<T>);

/// The decoded form of a [`TensorRow`] column: one flat typed buffer plus the stride to read it at.
///
/// Typed at decode time rather than per row. `FlatElements::row` re-derives its typed slice on every
/// call, which costs a ptype check and a buffer downcast per row; a row loop reads every row, so it
/// pays that once here instead.
pub struct TensorRows<T> {
    /// Every row's elements, back to back.
    elements: Buffer<T>,

    /// Number of logical tensor rows, stored so zero-width tensors retain their length.
    rows: usize,

    /// Elements per row, the length of each row slice.
    list_size: usize,

    /// `list_size` for a full column and `0` for constant-backed storage, so `index * stride` pins a
    /// constant to its single materialized row without a branch in the loop.
    stride: usize,
}

impl<T: Float + NativePType> InputElement for TensorRow<T> {
    type Column = TensorRows<T>;
    type Varying<'a> = &'a TensorRows<T>;
    type Elem<'a> = &'a [T];

    // Tensor storage is a fully materialized non-nullable primitive buffer, so the elements behind
    // a null row are arbitrary values rather than an unresolvable reference.
    const DENSE_SAFE: bool = true;
    // Tensor storage is a primitive buffer; reading it cannot fail on account of its values.
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let tensor_match = validate_tensor_float_input(dtype)?;
        let expected = T::PTYPE;
        vortex_ensure_eq!(
            tensor_match.element_ptype(),
            expected,
            "expected a tensor of {expected} elements, got {dtype}",
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let rows = array.len();
        let list_size = validate_tensor_float_input(array.dtype())?.list_size() as usize;
        let ext: ExtensionArray = array.execute(ctx)?;
        let flat = extract_flat_elements(ext.storage_array(), list_size, ctx)?;

        Ok(TensorRows {
            rows,
            list_size: flat.list_size(),
            stride: flat.row_stride(),
            elements: flat.into_buffer::<T>(),
        })
    }

    fn get(column: &Self::Column, index: usize) -> &[T] {
        let start = index * column.stride;
        &column.elements.as_slice()[start..start + column.list_size]
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        column
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        column.rows
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> &'a [T]
    where
        Self: 'a,
    {
        Self::get(column, index)
    }
}

/// Test-only probe recording which operands the last `prepare` step saw as batch-constant, so a
/// test can assert its inputs took the stride-0 decode path rather than merely producing the right
/// values through the varying path.
#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;

    thread_local! {
        /// Bitmask of the constant operands the last `prepare` saw (bit 0 for the lhs, bit 1 for
        /// the rhs). Thread-local rather than a process global so concurrent tests in one process
        /// (plain `cargo test`) cannot race it; execution runs on the calling thread.
        pub(crate) static SEEN_CONSTANTS: Cell<u8> = const { Cell::new(u8::MAX) };
    }

    /// Record which operands `prepare` saw as constant.
    pub(crate) fn record(lhs_constant: bool, rhs_constant: bool) {
        SEEN_CONSTANTS.set(u8::from(lhs_constant) | (u8::from(rhs_constant) << 1));
    }
}
