// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod fixed_size_list;
mod list_view;
mod null;
mod primitive;
mod struct_;
mod varbinview;

use bool::bool_uncompressed_size_in_bytes;
use decimal::decimal_uncompressed_size_in_bytes;
use extension::extension_uncompressed_size_in_bytes;
use fixed_size_list::fixed_size_list_uncompressed_size_in_bytes;
use list_view::list_view_uncompressed_size_in_bytes;
use null::null_uncompressed_size_in_bytes;
use primitive::primitive_uncompressed_size_in_bytes;
use struct_::struct_uncompressed_size_in_bytes;
use varbinview::varbinview_uncompressed_size_in_bytes;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::ConstantArray;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// Return the uncompressed size of an array in bytes.
///
/// See [`UncompressedSizeInBytes`] for details.
pub fn uncompressed_size_in_bytes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    let size = uncompressed_size_in_bytes_u64(array, ctx)?;

    usize::try_from(size)
        .map_err(|e| vortex_err!("Failed to convert uncompressed size to usize: {e}"))
}

fn uncompressed_size_in_bytes_u64(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    if let Some(Precision::Exact(size_scalar)) =
        array.statistics().get(Stat::UncompressedSizeInBytes)
    {
        return u64::try_from(&size_scalar)
            .map_err(|e| vortex_err!("Failed to convert uncompressed size stat to u64: {e}"));
    }

    let mut acc =
        Accumulator::try_new(UncompressedSizeInBytes, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    let size = result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("uncompressed_size_in_bytes result should not be null");

    array.statistics().set(
        Stat::UncompressedSizeInBytes,
        Precision::Exact(ScalarValue::from(size)),
    );

    Ok(size)
}

/// Sum the canonical, recursively **uncompressed** data size for an array.
///
/// Applies to all types and returns a non-null `u64`. Encoding kernels can return this aggregate
/// directly from metadata to avoid decoding arrays whose uncompressed size is known.
///
/// This is generally useful for various execution engines to pick better join orderings.
#[derive(Clone, Debug)]
pub struct UncompressedSizeInBytes;

impl AggregateFnVTable for UncompressedSizeInBytes {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.uncompressed_size_in_bytes")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("UncompressedSizeInBytes is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supports_uncompressed_size_in_bytes(input_dtype)
            .then_some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let size = other
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("uncompressed_size_in_bytes partial should not be null");
        *partial = partial
            .checked_add(size)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        *partial = 0;
    }

    #[inline]
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(Precision::Exact(size_scalar)) =
            batch.statistics().get(Stat::UncompressedSizeInBytes)
        else {
            return Ok(false);
        };

        let size = u64::try_from(&size_scalar)
            .map_err(|e| vortex_err!("Failed to convert uncompressed size stat to u64: {e}"))?;
        *partial = partial
            .checked_add(size)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        Ok(true)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let size = match batch {
            Columnar::Canonical(canonical) => canonical_uncompressed_size_in_bytes(canonical, ctx)?,
            Columnar::Constant(constant) => {
                let array = constant.clone().into_array();
                materialized_uncompressed_size_in_bytes(&array)
            }
        };
        *partial = partial
            .checked_add(size)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn canonical_uncompressed_size_in_bytes(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    match canonical {
        Canonical::Null(array) => Ok(null_uncompressed_size_in_bytes(array)),
        Canonical::Bool(array) => bool_uncompressed_size_in_bytes(array, ctx),
        Canonical::Primitive(array) => primitive_uncompressed_size_in_bytes(array, ctx),
        Canonical::Decimal(array) => decimal_uncompressed_size_in_bytes(array, ctx),
        Canonical::VarBinView(array) => varbinview_uncompressed_size_in_bytes(array, ctx),
        Canonical::List(array) => list_view_uncompressed_size_in_bytes(array, ctx),
        Canonical::FixedSizeList(array) => fixed_size_list_uncompressed_size_in_bytes(array, ctx),
        Canonical::Struct(array) => struct_uncompressed_size_in_bytes(array, ctx),
        Canonical::Extension(array) => extension_uncompressed_size_in_bytes(array, ctx),
        Canonical::Variant(_) => {
            vortex_bail!("UncompressedSizeInBytes is not supported for Variant arrays")
        }
    }
}

fn supports_uncompressed_size_in_bytes(dtype: &DType) -> bool {
    match dtype {
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            supports_uncompressed_size_in_bytes(element_dtype)
        }
        DType::Struct(fields, _) => fields
            .fields()
            .all(|field| supports_uncompressed_size_in_bytes(&field)),
        DType::Extension(ext_dtype) => {
            supports_uncompressed_size_in_bytes(ext_dtype.storage_dtype())
        }
        DType::Variant(_) => false,
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
    }
}

fn materialized_uncompressed_size_in_bytes(array: &ArrayRef) -> u64 {
    let mut builder = builder_with_capacity(array.dtype(), array.len());
    unsafe {
        builder.extend_from_array_unchecked(array);
    }
    builder.finish().nbytes()
}

fn validity_uncompressed_size_in_bytes(validity: Mask) -> VortexResult<u64> {
    match validity {
        Mask::AllTrue(_) => Ok(0),
        Mask::AllFalse(len) => Ok(ConstantArray::new(false, len).into_array().nbytes()),
        Mask::Values(values) => packed_bit_buffer_size_in_bytes(values.len()),
    }
}

fn packed_bit_buffer_size_in_bytes(len: usize) -> VortexResult<u64> {
    u64::try_from(len.div_ceil(8))
        .map_err(|e| vortex_err!("Failed to convert bit buffer length to u64: {e}"))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::materialized_uncompressed_size_in_bytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::VariantArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    fn aggregate(array: &ArrayRef) -> VortexResult<u64> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut acc =
            Accumulator::try_new(UncompressedSizeInBytes, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(array, &mut ctx)?;
        acc.finish()?
            .as_primitive()
            .typed_value::<u64>()
            .ok_or_else(|| vortex_err!("uncompressed size result should not be null"))
    }

    #[test]
    fn primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn nullable_primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn all_invalid_primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![0i32, 0, 0], Validity::AllInvalid).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true, true, false]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn nullable_bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn all_invalid_bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([None::<bool>, None, None]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn null_matches_materialized_size() -> VortexResult<()> {
        let array = NullArray::new(5).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn decimal_matches_materialized_size() -> VortexResult<()> {
        let array = DecimalArray::new(
            buffer![12345i64, -123i64, 0i64],
            DecimalDType::new(5, 2),
            Validity::NonNullable,
        )
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn varbinview_matches_materialized_size() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            None,
            Some("this string is longer than twelve bytes"),
        ])
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn list_matches_materialized_size() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let offsets = buffer![2u32, 0].into_array();
        let sizes = buffer![2u32, 1].into_array();
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn fixed_size_list_matches_materialized_size() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 2).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn struct_matches_materialized_size() -> VortexResult<()> {
        let ints = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let strings = VarBinViewArray::from_iter_nullable_str([Some("alpha"), None, Some("omega")])
            .into_array();
        let array = StructArray::try_new(
            FieldNames::from(["ints", "strings"]),
            vec![ints, strings],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn extension_matches_materialized_size() -> VortexResult<()> {
        let storage = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let ext_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
        let array = ExtensionArray::new(ext_dtype, storage).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn variant_stat_is_unsupported() -> VortexResult<()> {
        let child = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let array = VariantArray::new(child).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert_eq!(
            array
                .statistics()
                .compute_uncompressed_size_in_bytes(&mut ctx),
            None
        );
        Ok(())
    }

    #[test]
    fn constant_matches_materialized_size() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        Ok(())
    }

    #[test]
    fn chunked_sums_chunk_sizes() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        let chunk2 = PrimitiveArray::new(buffer![4i32, 5], Validity::NonNullable).into_array();
        let expected = materialized_uncompressed_size_in_bytes(&chunk1)
            + materialized_uncompressed_size_in_bytes(&chunk2);
        let chunked = ChunkedArray::try_new(
            vec![chunk1, chunk2],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();

        assert_eq!(aggregate(&chunked)?, expected);
        Ok(())
    }

    #[test]
    fn uses_cached_exact_stat() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();
        array.statistics().set(
            Stat::UncompressedSizeInBytes,
            Precision::Exact(ScalarValue::from(123u64)),
        );

        assert_eq!(aggregate(&array)?, 123);
        Ok(())
    }

    #[test]
    fn helper_caches_result() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let size = uncompressed_size_in_bytes(&array, &mut ctx)?;

        assert_eq!(
            array.statistics().get(Stat::UncompressedSizeInBytes),
            Some(Precision::exact(u64::try_from(size)?))
        );
        Ok(())
    }

    #[test]
    fn state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = UncompressedSizeInBytes.empty_partial(&EmptyOptions, &dtype)?;

        UncompressedSizeInBytes.combine_partials(
            &mut state,
            Scalar::primitive(5u64, Nullability::NonNullable),
        )?;
        UncompressedSizeInBytes.combine_partials(
            &mut state,
            Scalar::primitive(3u64, Nullability::NonNullable),
        )?;

        let result = UncompressedSizeInBytes.to_scalar(&state)?;
        UncompressedSizeInBytes.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(8));
        Ok(())
    }
}
