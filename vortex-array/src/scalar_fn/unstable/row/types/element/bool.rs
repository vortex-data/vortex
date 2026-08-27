// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::validity::Validity;

// SAFETY: the view is a bit buffer, and its reported length is the buffer length.
unsafe impl InputElement for bool {
    type Column = BitBuffer;
    type View<'a> = &'a BitBuffer;
    type Elem<'a> = bool;

    // Every bit of the buffer is readable, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_INFALLIBLE: bool = true;

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

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    fn get(column: &Self::Column, index: usize) -> bool {
        column.value(index)
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> bool
    where
        Self: 'a,
    {
        view.value(index)
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> bool
    where
        Self: 'a,
    {
        // SAFETY: forwarded from this method's contract.
        unsafe { view.value_unchecked(index) }
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

    fn build_from<S, F>(source: S, apply: F) -> ArrayRef
    where
        S: IndexedSource,
        F: Fn(S::Item) -> Self,
    {
        let len = source.len();
        let values = BitBuffer::collect_bool(len, |index| {
            // SAFETY: `collect_bool` only invokes this closure with `index < len`, and
            // `len` is `source.len()`.
            apply(unsafe { source.get_unchecked(index) })
        });

        BoolArray::new(values, Validity::NonNullable).into_array()
    }
}
