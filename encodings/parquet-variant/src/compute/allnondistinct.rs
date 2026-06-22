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

        let identical = match (lhs.typed_value_array(), rhs.typed_value_array()) {
            (Some(lhs_typed), Some(rhs_typed)) => all_non_distinct(lhs_typed, rhs_typed, ctx)?,
            _ => match (lhs.value_array(), rhs.value_array()) {
                (Some(lhs_value), Some(rhs_value)) => all_non_distinct(lhs_value, rhs_value, ctx)?,
                // Mixed shredding layouts: let the generic canonical path handle it.
                _ => return Ok(None),
            },
        };

        Ok(Some(Scalar::bool(identical, Nullability::NonNullable)))
    }
}
