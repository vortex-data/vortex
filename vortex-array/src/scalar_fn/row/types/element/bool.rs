// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::InputElement;
use crate::scalar_fn::OutputElement;
use crate::validity::Validity;

// SAFETY: the varying view is a bit buffer, and its reported length is the buffer length.
unsafe impl InputElement for bool {
    type Column = BitBuffer;
    type Varying<'a> = &'a BitBuffer;
    type Elem<'a> = bool;

    // Every bit of the buffer is readable, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Bool(_)),
            "expected a Bool column, got {dtype}",
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(array.execute::<BoolArray>(ctx)?.into_bit_buffer())
    }

    fn get(column: &Self::Column, index: usize) -> bool {
        column.value(index)
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        column
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        column.len()
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> bool
    where
        Self: 'a,
    {
        column.value(index)
    }

    unsafe fn get_varying_unchecked<'a>(column: &Self::Varying<'a>, index: usize) -> bool
    where
        Self: 'a,
    {
        // SAFETY: forwarded from this method's contract.
        unsafe { column.value_unchecked(index) }
    }
}

impl OutputElement for bool {
    fn element_dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        // `From<Vec<bool>>` uses the bulk bit-packing path.
        BoolArray::new(BitBuffer::from(values), Validity::NonNullable).into_array()
    }
}
