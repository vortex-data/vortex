// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::validity::Validity;

// SAFETY: the view is a native slice, and its reported length is the slice length.
unsafe impl<T: NativePType> InputElement for T {
    type Column = Buffer<T>;
    type View<'a> = &'a [T];
    type Elem<'a> = T;

    // Every lane of the buffer holds a `T`, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let expected = T::PTYPE;
        let DType::Primitive(ptype, _) = dtype else {
            vortex_bail!("expected a {expected} column, got {dtype}");
        };
        vortex_ensure_eq!(
            *ptype,
            expected,
            "expected a {expected} column, got {dtype}"
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(array.execute::<PrimitiveArray>(ctx)?.into_buffer::<T>())
    }

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    fn get(column: &Self::Column, index: usize) -> T {
        column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> T
    where
        Self: 'a,
    {
        view[index]
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> T
    where
        Self: 'a,
    {
        // SAFETY: forwarded from this method's contract.
        unsafe { *view.get_unchecked(index) }
    }
}

impl<T: NativePType> OutputElement for T {
    fn element_dtype() -> DType {
        DType::Primitive(T::PTYPE, Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }
}
