// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tensor row inputs for scalar functions.
//!
//! [`TensorRow`] decodes one flat row from either an ordinary tensor column or either supported
//! batch-constant representation. Batch execution owns input validity, so decoding handles only
//! the non-null coordinate storage.

use std::marker::PhantomData;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::ViewLen;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// Returns the common float element type of tensor row-function arguments.
pub fn tensor_element_ptype(args: &[DType]) -> VortexResult<PType> {
    let (first, rest) = args
        .split_first()
        .ok_or_else(|| vortex_err!("tensor row function requires at least one input"))?;

    for argument in rest {
        vortex_ensure!(
            first.eq_ignore_nullability(argument),
            "tensor row-function inputs must have the same dtype, got {first} and {argument}",
        );
    }

    Ok(validate_tensor_float_input(first)?.element_ptype())
}

/// A tensor-valued row-function input whose element is one flat tensor row.
pub struct TensorRow<T>(PhantomData<T>);

/// A decoded tensor column with constant-width rows.
pub struct TensorRows<T> {
    elements: Buffer<T>,
    row_count: usize,
    row_width: usize,
    stride: usize,
}

impl<T> ViewLen for TensorRows<T> {
    fn len(&self) -> usize {
        self.row_count
    }
}

fn decode_tensor_storage<T: NativePType>(
    storage: &ArrayRef,
    row_count: usize,
    row_width: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TensorRows<T>> {
    let flat = extract_flat_elements(storage, row_width, ctx)?;
    let row_width = flat.list_size();
    let stride = flat.row_stride();
    let elements = flat.into_buffer::<T>();

    let expected_elements = if stride == 0 {
        row_width
    } else {
        vortex_ensure_eq!(
            stride,
            row_width,
            "per-row tensor stride must equal its width, got {stride}",
        );
        let Some(expected_elements) = row_count.checked_mul(stride) else {
            vortex_bail!(
                "tensor row storage length must fit usize, got {row_count} rows of width {stride}",
            );
        };
        expected_elements
    };
    vortex_ensure_eq!(
        elements.len(),
        expected_elements,
        "tensor row storage must contain {expected_elements} elements, got {}",
        elements.len(),
    );

    Ok(TensorRows {
        elements,
        row_count,
        row_width,
        stride,
    })
}

// SAFETY: `TensorRows` records the row count validated during decode, and both checked and
// unchecked access use the same stride and row width.
unsafe impl<T: Float + NativePType> InputElement for TensorRow<T> {
    type Column = TensorRows<T>;
    type Constant = Buffer<T>;
    type View<'a> = &'a TensorRows<T>;
    type Elem<'a> = &'a [T];

    const DENSE_SAFE: bool = true;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let tensor_match = validate_tensor_float_input(dtype)?;
        let expected_element_ptype = T::PTYPE;
        vortex_ensure_eq!(
            tensor_match.element_ptype(),
            expected_element_ptype,
            "tensor row input must use {expected_element_ptype} elements, got {dtype}",
        );

        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        // Batch execution owns the mask and restores it on the result.
        let array = match array.as_opt::<Masked>() {
            Some(masked) => masked.child().clone(),
            None => array,
        };

        let row_count = array.len();
        let row_width = validate_tensor_float_input(array.dtype())?.list_size() as usize;
        let extension: ExtensionArray = array.execute(ctx)?;
        decode_tensor_storage(extension.storage_array(), row_count, row_width, ctx)
    }

    fn decode_constant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Constant> {
        let row_width = validate_tensor_float_input(array.dtype())?.list_size() as usize;
        let storage = if let Some(constant) = array.as_opt::<Constant>() {
            let scalar = constant.scalar().as_extension().to_storage_scalar();
            ConstantArray::new(scalar, 1).into_array()
        } else if let Some(extension) = array.as_opt::<Extension>() {
            extension.storage_array().clone()
        } else {
            vortex_bail!(
                "a tensor batch constant must use the Constant encoding or constant extension \
                 storage, got {}",
                array.encoding_id()
            );
        };

        let decoded = decode_tensor_storage::<T>(&storage, 1, row_width, ctx)?;
        vortex_ensure_eq!(
            decoded.elements.len(),
            row_width,
            "decoded tensor constant must contain {row_width} elements, got {}",
            decoded.elements.len(),
        );

        Ok(decoded.elements)
    }

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    fn get(column: &Self::Column, index: usize) -> &[T] {
        let start = index * column.stride;
        &column.elements.as_slice()[start..start + column.row_width]
    }

    fn get_constant(constant: &Self::Constant) -> &[T] {
        constant.as_slice()
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> &'a [T]
    where
        Self: 'a,
    {
        Self::get(view, index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> &'a [T]
    where
        Self: 'a,
    {
        let start = index * view.stride;

        // SAFETY: decode established one complete stored row for stride 0, or `row_count`
        // contiguous `row_width`-element rows otherwise. The caller guarantees
        // `index < row_count`.
        unsafe {
            std::slice::from_raw_parts(view.elements.as_slice().as_ptr().add(start), view.row_width)
        }
    }
}
