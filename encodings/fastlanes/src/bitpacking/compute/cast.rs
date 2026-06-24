// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::bitpacking::BitPacked;
use crate::bitpacking::array::BitPackedArrayExt;
use crate::bitpacking::array::bitpack_decompress::unpack_map_into_builder;

/// Returns `true` if casting `src` to `tgt` is a widening integer cast for which every value a
/// bit-packed array can hold is guaranteed to be representable in `tgt` (so no per-value bounds
/// check is needed). This holds when `tgt` is strictly wider and either the source is unsigned
/// (always non-negative, fits in any wider type) or the target is also signed (sign-extension).
fn is_widening_int_cast(src: PType, tgt: PType) -> bool {
    src.is_int()
        && tgt.is_int()
        && tgt.byte_width() > src.byte_width()
        && (src.is_unsigned_int() || tgt.is_signed_int())
}

fn build_with_validity(
    array: ArrayView<'_, BitPacked>,
    dtype: &DType,
    new_validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(BitPacked::try_new(
        array.packed().clone(),
        dtype.as_ptype(),
        new_validity,
        array
            .patches()
            .map(|patches| patches.map_values(|values| values.cast(dtype.clone())))
            .transpose()?,
        array.bit_width(),
        array.len(),
        array.offset(),
    )?
    .into_array())
}

impl CastReduce for BitPacked {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }
        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        build_with_validity(array, dtype, new_validity).map(Some)
    }
}

impl CastKernel for BitPacked {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Nullability-only change: keep the values bit-packed, just adjust validity.
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity =
                array
                    .validity()?
                    .cast_nullability(dtype.nullability(), array.len(), ctx)?;
            return build_with_validity(array, dtype, new_validity).map(Some);
        }

        // Widening integer cast: unpack each FastLanes chunk into a cache-resident scratch buffer
        // and cast-copy straight into the wide output, avoiding a full-length intermediate buffer
        // and the generic cast kernel's bounds-check scan (unnecessary when widening).
        let DType::Primitive(tgt, tgt_nullability) = dtype else {
            return Ok(None);
        };
        let (tgt, tgt_nullability) = (*tgt, *tgt_nullability);
        let src = array.dtype().as_ptype();
        if !is_widening_int_cast(src, tgt) {
            return Ok(None);
        }

        // Surface the standard error if a nullable source with nulls is cast to a non-nullable
        // type; on success the per-value validity is handled inside the unpack below.
        array
            .validity()?
            .cast_nullability(tgt_nullability, array.len(), ctx)?;

        let result = match_each_integer_ptype!(tgt, |T| {
            let mut builder = PrimitiveBuilder::<T>::with_capacity(tgt_nullability, array.len());
            match_each_integer_ptype!(src, |F| {
                unpack_map_into_builder::<F, T, _>(array, &mut builder, ctx, |v: F| v.as_())?;
            });
            builder.finish_into_primitive().into_array()
        });
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::match_each_integer_ptype;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedArray;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn bp(array: &ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(array, bit_width, &mut SESSION.create_execution_ctx()).unwrap()
    }

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed = bp(&buffer![10u8, 20, 30, 40, 50, 60].into_array(), 6);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([10u32, 20, 30, 40, 50, 60]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_cast_bitpacked_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(5u16), None, Some(10), Some(15), None]);
        let packed = bp(&values.into_array(), 4);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    /// End-to-end check that the real engine path `array.cast(target).execute()` routes through the
    /// bit-packed widening pushdown and matches a plain primitive cast over the same values, across
    /// every supported integer pair, several chunk-boundary lengths, and a sliced (offset > 0) case.
    #[test]
    fn test_cast_bitpacked_widening_via_execute() -> VortexResult<()> {
        fn values<T: NativePType>(len: usize) -> PrimitiveArray {
            PrimitiveArray::from_iter((0..len).map(|i| {
                let value = if i % 17 == 0 { 31 } else { i % 8 };
                <T as num_traits::FromPrimitive>::from_usize(value)
                    .expect("test values fit every integer ptype")
            }))
        }

        fn supported(src: PType, tgt: PType) -> bool {
            src.is_int()
                && tgt.is_int()
                && tgt.byte_width() > src.byte_width()
                && (src.is_unsigned_int() || tgt.is_signed_int())
        }

        let ptypes = [
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
        ];
        // Lengths exercise empty, sub-chunk, exact chunk, chunk+1, and multi-chunk-with-trailer.
        let lengths = [0, 1, 7, 1023, 1024, 1025, 2051];

        for src in ptypes {
            for tgt in ptypes {
                if !supported(src, tgt) {
                    continue;
                }

                for len in lengths {
                    let source = match_each_integer_ptype!(src, |S| { values::<S>(len) });
                    let source_ref = source.into_array();
                    let target = DType::Primitive(tgt, Nullability::NonNullable);
                    let mut ctx = SESSION.create_execution_ctx();

                    // Reference: plain primitive cast of the same values.
                    let reference = source_ref
                        .clone()
                        .cast(target.clone())?
                        .execute::<PrimitiveArray>(&mut ctx)?;

                    // Candidate: bit-pack, then cast through the real engine. This dispatches to
                    // `BitPacked`'s `CastKernel` widening pushdown.
                    let packed = bp(&source_ref, 3).into_array();
                    let casted = packed
                        .cast(target.clone())?
                        .execute::<PrimitiveArray>(&mut ctx)?;
                    assert_arrays_eq!(casted, reference, &mut ctx);

                    // Also exercise the sliced/offset path (offset > 0, trailer present).
                    if len >= 4 {
                        let lo = len / 4;
                        let hi = len - len / 4;
                        let sliced = bp(&source_ref, 3).into_array().slice(lo..hi)?;
                        let casted = sliced
                            .cast(target.clone())?
                            .execute::<PrimitiveArray>(&mut ctx)?;
                        let reference = source_ref
                            .clone()
                            .slice(lo..hi)?
                            .cast(target.clone())?
                            .execute::<PrimitiveArray>(&mut ctx)?;
                        assert_arrays_eq!(casted, reference, &mut ctx);
                    }
                }
            }
        }

        Ok(())
    }

    #[rstest]
    #[case(bp(&buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array(), 6))]
    #[case(bp(&buffer![0u16, 100, 200, 300, 400, 500].into_array(), 9))]
    #[case(bp(&buffer![0u32, 1000, 2000, 3000, 4000].into_array(), 12))]
    #[case(bp(&PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]).into_array(), 4))]
    fn test_cast_bitpacked_conformance(#[case] array: BitPackedArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
