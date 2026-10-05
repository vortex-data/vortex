// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for lazy casts that widen primitive values.
//!
//! The child streams up and each chunk is converted while L1-resident, straight into the output
//! when materializing, so neither the child's values nor a second full-length buffer is written.
//! Only casts that cannot fail stream: a lossless widening that keeps nulls representable. Others
//! check their values against the target range, which the level-wise kernel does.

use std::marker::PhantomData;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::ScalarFn;
use crate::arrays::primitive::compute::cast::casts_losslessly_to;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;
use crate::chunk_iter::ScratchChunk;
use crate::chunk_iter::ValueType;
use crate::chunk_iter::emit_with;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::Cast;

pub(super) fn decompress_chunks_type(array: ArrayView<'_, ScalarFn>) -> Option<ValueType> {
    ValueType::primitive(array.dtype())
        .filter(|_| widened_child(&array).is_some_and(|child| child.supports_decompress_chunks()))
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, ScalarFn>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let child =
        widened_child(&array).ok_or_else(|| vortex_err!("only lossless primitive casts stream"))?;
    let (from, to) = (child.dtype().as_ptype(), array.dtype().as_ptype());
    if from == to {
        return child.decompress_child_chunks(ctx, sink);
    }
    match_each_native_ptype!(from, |F| {
        match_each_native_ptype!(to, |T| {
            let mut adapter = WidenSink::<F, T> {
                scratch: ScratchChunk::new(),
                inner: sink,
                _from: PhantomData,
            };
            child.decompress_child_chunks(ctx, &mut adapter)
        })
    })
}

/// The child of a cast that converts every primitive value losslessly and only ever widens the
/// nullability, so it cannot fail.
fn widened_child<'a>(array: &'a ArrayView<'_, ScalarFn>) -> Option<&'a ArrayRef> {
    let DType::Primitive(to, to_nullability) = array.scalar_fn().as_opt::<Cast>()? else {
        return None;
    };
    let child = array.get_child(0);
    let DType::Primitive(from, from_nullability) = child.dtype() else {
        return None;
    };
    (casts_losslessly_to(*from, *to)
        && (to_nullability.is_nullable() || !from_nullability.is_nullable()))
    .then_some(child)
}

/// Converts each chunk of `F` values from the child to `T` and forwards it.
struct WidenSink<'a, F, T> {
    scratch: ScratchChunk<T>,
    inner: &'a mut dyn ChunkSink,
    _from: PhantomData<F>,
}

impl<F, T> ChunkSink for WidenSink<'_, F, T>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let values = chunk.as_slice::<F>();
        emit_with(&mut *self.inner, T::PTYPE, rows, &mut self.scratch, |out| {
            for (out, &value) in out.iter_mut().zip(values) {
                *out = value.as_();
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::test_harness::assert_streams_like_execute;

    fn nullable_u16s() -> ArrayRef {
        PrimitiveArray::from_option_iter((0..3000u16).map(|i| (i % 7 != 3).then_some(i * 19)))
            .into_array()
    }

    #[rstest]
    #[case::widen(PType::I64, Nullability::Nullable)]
    #[case::to_float(PType::F64, Nullability::Nullable)]
    #[case::same_ptype(PType::U16, Nullability::Nullable)]
    fn cast_streams_like_execute(
        #[case] ptype: PType,
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(ptype, nullability);
        let cast = Cast::new(nullable_u16s(), dtype.clone()).into_array();
        assert!(cast.supports_decompress_chunks());
        assert_streams_like_execute(&cast, &mut ctx)?;
        // Over a lazy slice, as scans leave the child.
        let sliced = SliceArray::new(nullable_u16s(), 517..2900).into_array();
        assert_streams_like_execute(&Cast::new(sliced, dtype).into_array(), &mut ctx)
    }

    /// Casts that may fail on a value, or on a null, keep the level-wise kernel's checks.
    #[rstest]
    #[case::narrowing(PType::U8, Nullability::Nullable)]
    #[case::sign_change(PType::I16, Nullability::Nullable)]
    #[case::drops_nullability(PType::I64, Nullability::NonNullable)]
    fn fallible_casts_do_not_stream(#[case] ptype: PType, #[case] nullability: Nullability) {
        let cast = Cast::new(nullable_u16s(), DType::Primitive(ptype, nullability)).into_array();
        assert!(!cast.supports_decompress_chunks());
    }
}
