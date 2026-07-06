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
use crate::ParquetVariantArrayExt;

/// Lets `AllNonDistinct` compare two `ParquetVariant` arrays without canonicalizing them.
///
/// `AllNonDistinct` accumulates over a `Struct{lhs, rhs}` batch, so this kernel is registered for
/// the struct encoding and inspects the two children. When both are `ParquetVariant`, we compare
/// the typed (`typed_value`) arrays if both sides are shredded, and fall back to the raw `value`
/// arrays otherwise. Comparing these child arrays directly avoids re-canonicalizing the variant
/// (which would recurse through the `Variant` canonical form).
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
        let lhs = batch.unmasked_field(0);
        let rhs = batch.unmasked_field(1);
        let (Some(lhs), Some(rhs)) = (
            lhs.as_opt::<ParquetVariant>(),
            rhs.as_opt::<ParquetVariant>(),
        ) else {
            return Ok(None);
        };

        let typed_identical = match (lhs.typed_value_array(), rhs.typed_value_array()) {
            (Some(lhs_typed), Some(rhs_typed)) => {
                if lhs_typed.dtype().eq_ignore_nullability(rhs_typed.dtype()) {
                    all_non_distinct(lhs_typed, rhs_typed, ctx)?
                } else {
                    return Ok(None);
                }
            }
            _ => true,
        };

        if typed_identical {
            let values_identical = match (lhs.value_array(), rhs.value_array()) {
                (Some(lhs_value), Some(rhs_value)) => all_non_distinct(lhs_value, rhs_value, ctx)?,
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

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ParquetVariant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
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
}
