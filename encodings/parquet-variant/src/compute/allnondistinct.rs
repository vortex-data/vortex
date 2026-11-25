// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Struct;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::ParquetVariant;
use crate::ParquetVariantArraySlotsExt;

/// Lets `AllNonDistinct` compare two `ParquetVariant` arrays without canonicalizing them.
///
/// `AllNonDistinct` accumulates over a `Struct{lhs, rhs}` batch, so this kernel is registered for
/// the struct encoding and inspects the two children. When both are `ParquetVariant`, we first
/// check batch and outer variant validity, then require identical `metadata` dictionaries (falling
/// back to the canonical `Variant` path otherwise) before comparing `typed_value` and `value`
/// children directly.
#[derive(Debug)]
pub struct AllNonDistinctParquetVariant;

impl DynAggregateKernel for AllNonDistinctParquetVariant {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<AllNonDistinct>() {
            return Ok(None);
        }

        let Some(batch) = batch.as_opt::<Struct>() else {
            return Ok(None);
        };
        if batch.is_empty() {
            return Ok(Some(Scalar::bool(true, Nullability::NonNullable)));
        }

        let lhs = batch.unmasked_field(0);
        let rhs = batch.unmasked_field(1);
        let (Some(lhs), Some(rhs)) = (
            lhs.as_opt::<ParquetVariant>(),
            rhs.as_opt::<ParquetVariant>(),
        ) else {
            return Ok(None);
        };

        // Respect both the batch struct mask (rows masked out by the caller) and each side's
        // outer variant validity. A valid batch row where only one side is null is distinct, and
        // rows that are null on both sides must be excluded before comparing physical children.
        let batch_validity = batch.validity()?;
        let lhs_validity = lhs.validity()?;
        let rhs_validity = rhs.validity()?;
        let active_mask = if batch_validity.definitely_no_nulls()
            && lhs_validity.definitely_no_nulls()
            && rhs_validity.definitely_no_nulls()
        {
            None
        } else {
            let batch_mask = batch_validity.execute_mask(batch.len(), ctx)?;
            if batch_mask.true_count() == 0 {
                return Ok(Some(Scalar::bool(true, Nullability::NonNullable)));
            }

            let lhs_mask = &batch_mask & &lhs_validity.execute_mask(lhs.len(), ctx)?;
            let rhs_mask = &batch_mask & &rhs_validity.execute_mask(rhs.len(), ctx)?;
            if lhs_mask != rhs_mask {
                return Ok(Some(Scalar::bool(false, Nullability::NonNullable)));
            }
            if lhs_mask.true_count() == 0 {
                return Ok(Some(Scalar::bool(true, Nullability::NonNullable)));
            }
            (lhs_mask.true_count() < batch.len()).then_some(lhs_mask)
        };

        let filter_child = |child: &ArrayRef| -> VortexResult<ArrayRef> {
            match &active_mask {
                Some(mask) => child.filter(mask.clone()),
                None => Ok(child.clone()),
            }
        };

        // Comparing raw `value` bytes is only meaningful when both sides share identical metadata
        // dictionaries. Decline to the canonical path when metadata differs, since logically equal
        // variants may also use differently ordered dictionaries.
        let lhs_metadata = filter_child(lhs.metadata())?;
        let rhs_metadata = filter_child(rhs.metadata())?;
        if !all_non_distinct(&lhs_metadata, &rhs_metadata, ctx)? {
            return Ok(None);
        }

        let typed_identical = match (lhs.typed_value(), rhs.typed_value()) {
            (Some(lhs_typed), Some(rhs_typed)) => {
                // `all_non_distinct` requires equal dtypes, including nullability.
                if lhs_typed.dtype() == rhs_typed.dtype() {
                    let lhs_typed = filter_child(lhs_typed)?;
                    let rhs_typed = filter_child(rhs_typed)?;
                    all_non_distinct(&lhs_typed, &rhs_typed, ctx)?
                } else {
                    return Ok(None);
                }
            }
            (None, None) => true,
            // Mixed shredding layouts: let the generic canonical path handle it.
            _ => return Ok(None),
        };

        if typed_identical {
            let values_identical = match (lhs.value(), rhs.value()) {
                (Some(lhs_value), Some(rhs_value)) if lhs_value.dtype() == rhs_value.dtype() => {
                    let lhs_value = filter_child(lhs_value)?;
                    let rhs_value = filter_child(rhs_value)?;
                    all_non_distinct(&lhs_value, &rhs_value, ctx)?
                }
                // Value children that differ in nullability: let the generic canonical path
                // handle it.
                (Some(_), Some(_)) => return Ok(None),
                (None, None) => true,
                // Mixed shredding layouts: let the generic canonical path handle it.
                _ => return Ok(None),
            };
            Ok(Some(Scalar::bool(
                values_identical,
                Nullability::NonNullable,
            )))
        } else {
            Ok(Some(Scalar::bool(false, Nullability::NonNullable)))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use parquet_variant::VariantBuilder;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
    use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::AllNonDistinctParquetVariant;
    use crate::ParquetVariant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Non-nullable, minimally-valid metadata column of `len` rows.
    fn metadata(len: usize) -> ArrayRef {
        VarBinViewArray::from_iter_bin(vec![b"\x01\x00"; len]).into_array()
    }

    /// Non-nullable binary `value` column.
    fn binary<T: AsRef<[u8]>>(values: impl IntoIterator<Item = T>) -> ArrayRef {
        VarBinViewArray::from_iter_bin(values).into_array()
    }

    fn parquet_variant(
        len: usize,
        value: Option<ArrayRef>,
        typed_value: Option<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        Ok(
            ParquetVariant::try_new(Validity::NonNullable, metadata(len), value, typed_value)?
                .into_array(),
        )
    }

    #[test]
    fn all_non_distinct_matches_equal_unshredded() -> VortexResult<()> {
        let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
        let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_detects_distinct_unshredded() -> VortexResult<()> {
        let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
        let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x12"])), None)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_matches_equal_value_and_typed() -> VortexResult<()> {
        let typed = || buffer![1i32, 2].into_array();
        let lhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), Some(typed()))?;
        let rhs = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), Some(typed()))?;
        let mut ctx = SESSION.create_execution_ctx();
        assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_empty_is_true() -> VortexResult<()> {
        let lhs = parquet_variant(0, Some(binary(Vec::<&[u8]>::new())), None)?;
        let rhs = parquet_variant(0, Some(binary(Vec::<&[u8]>::new())), None)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_declines_when_only_one_side_is_shredded() -> VortexResult<()> {
        let shredded = parquet_variant(
            2,
            Some(binary([b"\x10", b"\x11"])),
            Some(buffer![1i32, 2].into_array()),
        )?;
        let unshredded = parquet_variant(2, Some(binary([b"\x10", b"\x11"])), None)?;
        let batch = StructArray::try_new(
            FieldNames::from(["lhs", "rhs"]),
            vec![shredded, unshredded],
            2,
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let aggregate = AllNonDistinct.bind(EmptyOptions);
        let result = AllNonDistinctParquetVariant.aggregate(&aggregate, &batch, &mut ctx)?;
        assert!(
            result.is_none(),
            "a shredded value cannot be compared to an unshredded one from `value` alone, so the \
             kernel must decline rather than report them identical; got {result:?}"
        );
        Ok(())
    }

    fn single_field_object(field: &str, value: i32) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        let mut obj = builder.new_object();
        obj.insert(field, value);
        obj.finish();
        builder.finish()
    }

    #[test]
    fn all_non_distinct_detects_different_metadata_dictionaries() -> VortexResult<()> {
        let (meta_a, val_a) = single_field_object("a", 1);
        let (meta_b, val_b) = single_field_object("b", 1);
        // Both objects reference field ID 0 with value 1, so their value bytes are identical
        // while their metadata dictionaries differ.
        assert_eq!(val_a, val_b);
        assert_ne!(meta_a, meta_b);

        let lhs = ParquetVariant::try_new(
            Validity::NonNullable,
            binary([meta_a.as_slice()]),
            Some(binary([val_a.as_slice()])),
            None,
        )?
        .into_array();
        let rhs = ParquetVariant::try_new(
            Validity::NonNullable,
            binary([meta_b.as_slice()]),
            Some(binary([val_b.as_slice()])),
            None,
        )?
        .into_array();

        let batch = StructArray::try_new(
            FieldNames::from(["lhs", "rhs"]),
            vec![lhs.clone(), rhs.clone()],
            1,
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let aggregate = AllNonDistinct.bind(EmptyOptions);
        assert!(
            AllNonDistinctParquetVariant
                .aggregate(&aggregate, &batch, &mut ctx)?
                .is_none(),
            "kernel must decline when metadata differs"
        );
        assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_matches_logically_equal_with_different_metadata() -> VortexResult<()> {
        let (meta_a, val_a) = single_field_object("a", 1);
        // Same logical object `{"a": 1}` with dictionary `["b", "a"]` (version=1, unsorted,
        // dict_size=2, offsets=[0, 1, 2], strings="ba"), so `"a"` is field ID 1 instead of 0.
        let meta_ba: &[u8] = b"\x01\x02\x00\x01\x02ba";
        let mut val_a_id1 = val_a.clone();
        // Single-field object layout: [header, num_elements=1, field_id, offset_0, offset_1, ...].
        assert_eq!(val_a_id1[2], 0);
        val_a_id1[2] = 1;

        let lhs = ParquetVariant::try_new(
            Validity::NonNullable,
            binary([meta_a.as_slice()]),
            Some(binary([val_a.as_slice()])),
            None,
        )?
        .into_array();
        let rhs = ParquetVariant::try_new(
            Validity::NonNullable,
            binary([meta_ba]),
            Some(binary([val_a_id1.as_slice()])),
            None,
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            lhs.execute_scalar(0, &mut ctx)?,
            rhs.execute_scalar(0, &mut ctx)?
        );
        assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_accumulator_detects_differing_outer_validity() -> VortexResult<()> {
        let (meta, val) = VariantBuilder::new().with_value(42i32).finish();
        let lhs = ParquetVariant::try_new(
            Validity::from_iter([true]),
            binary([meta.as_slice()]),
            Some(binary([val.as_slice()])),
            None,
        )?
        .into_array();
        let rhs = ParquetVariant::try_new(
            Validity::from_iter([false]),
            binary([meta.as_slice()]),
            Some(binary([val.as_slice()])),
            None,
        )?
        .into_array();

        let batch = StructArray::try_new(
            FieldNames::from(["lhs", "rhs"]),
            vec![lhs.clone(), rhs.clone()],
            1,
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
        acc.accumulate(&batch, &mut ctx)?;
        let result = acc.finish()?;
        assert_eq!(result.as_bool().value(), Some(false));
        assert!(!all_non_distinct(&lhs, &rhs, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn all_non_distinct_ignores_garbage_under_shared_outer_nulls() -> VortexResult<()> {
        let (meta_a, val_a) = single_field_object("a", 1);
        let (meta_b, val_b) = single_field_object("b", 2);
        let validity = Validity::from_iter([true, false]);

        let lhs = ParquetVariant::try_new(
            validity.clone(),
            binary([meta_a.as_slice(), meta_a.as_slice()]),
            Some(binary([val_a.as_slice(), val_a.as_slice()])),
            None,
        )?
        .into_array();
        let rhs = ParquetVariant::try_new(
            validity,
            binary([meta_a.as_slice(), meta_b.as_slice()]),
            Some(binary([val_a.as_slice(), val_b.as_slice()])),
            None,
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert!(all_non_distinct(&lhs, &rhs, &mut ctx)?);

        let batch = StructArray::try_new(
            FieldNames::from(["lhs", "rhs"]),
            vec![lhs, rhs],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
        acc.accumulate(&batch, &mut ctx)?;
        assert_eq!(acc.finish()?.as_bool().value(), Some(true));
        Ok(())
    }

    #[test]
    fn all_non_distinct_respects_batch_struct_validity() -> VortexResult<()> {
        let (meta, val) = VariantBuilder::new().with_value(42i32).finish();
        let lhs = ParquetVariant::try_new(
            Validity::from_iter([true, true]),
            binary([meta.as_slice(), meta.as_slice()]),
            Some(binary([val.as_slice(), val.as_slice()])),
            None,
        )?
        .into_array();
        let rhs = ParquetVariant::try_new(
            Validity::from_iter([true, false]),
            binary([meta.as_slice(), meta.as_slice()]),
            Some(binary([val.as_slice(), val.as_slice()])),
            None,
        )?
        .into_array();

        // Row 1 is masked out by the outer StructArray validity, so only row 0 counts.
        let batch = StructArray::try_new(
            FieldNames::from(["lhs", "rhs"]),
            vec![lhs, rhs],
            2,
            Validity::from_iter([true, false]),
        )?
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
        acc.accumulate(&batch, &mut ctx)?;
        assert_eq!(acc.finish()?.as_bool().value(), Some(true));
        Ok(())
    }
}
