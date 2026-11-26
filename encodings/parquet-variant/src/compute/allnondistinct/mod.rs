// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use std::fmt::Debug;

use parquet_variant::Variant as PqVariant;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::RepeatedArrayProbe;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::aggregate_fn::fns::all_non_distinct::variant_scalars_equal;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ParquetVariant;
use crate::ParquetVariantArraySlotsExt;

/// Lets `AllNonDistinct` compare two `ParquetVariant` arrays without canonicalizing them.
///
/// `AllNonDistinct` accumulates over a `Struct{lhs, rhs}` batch, so this kernel is registered for
/// the struct encoding and inspects the two children. When both are `ParquetVariant` with matching
/// shredding layouts and child dtypes, it checks batch and outer variant validity, then compares
/// `typed_value` and `value` children directly when `metadata` dictionaries match (or are unused
/// for fully shredded leaf values). When `metadata` is needed and differs, it compares active rows
/// directly via `ParquetVariant` scalars so raw-only nested fields are preserved.
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

        let lhs_arr = batch.unmasked_field(0);
        let rhs_arr = batch.unmasked_field(1);
        let (Some(lhs), Some(rhs)) = (
            lhs_arr.as_opt::<ParquetVariant>(),
            rhs_arr.as_opt::<ParquetVariant>(),
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

        // Check shredding layout and child dtype compatibility before comparing metadata or
        // children. When layouts or child dtypes differ (e.g. only one side is shredded, only one
        // side has a root `value`, or nullability differs), compare via `ParquetVariant` scalars
        // rather than declining to `Canonical::Variant`, which omits raw-only nested wrappers.
        let typed_pair = match (lhs.typed_value(), rhs.typed_value()) {
            (Some(lhs_typed), Some(rhs_typed)) if lhs_typed.dtype() == rhs_typed.dtype() => {
                Some((lhs_typed, rhs_typed))
            }
            (None, None) => None,
            _ => {
                let equal = compare_active_scalars(lhs_arr, rhs_arr, active_mask.as_ref(), ctx)?;
                return Ok(Some(Scalar::bool(equal, Nullability::NonNullable)));
            }
        };

        let value_pair = match (lhs.value(), rhs.value()) {
            (Some(lhs_value), Some(rhs_value)) if lhs_value.dtype() == rhs_value.dtype() => {
                Some((lhs_value, rhs_value))
            }
            (None, None) => None,
            _ => {
                let equal = compare_active_scalars(lhs_arr, rhs_arr, active_mask.as_ref(), ctx)?;
                return Ok(Some(Scalar::bool(equal, Nullability::NonNullable)));
            }
        };

        let filter_child = |child: &ArrayRef| -> VortexResult<ArrayRef> {
            match &active_mask {
                Some(mask) => child.filter(mask.clone()),
                None => Ok(child.clone()),
            }
        };

        // Metadata dictionaries only affect decoding when raw `value` bytes exist at the root or
        // may exist inside nested shredded struct/list wrappers. Fully shredded leaf values do not
        // read `metadata` and can always use bulk child comparison.
        let needs_metadata = value_pair.is_some()
            || typed_pair.is_some_and(|(lhs_typed, _)| {
                matches!(lhs_typed.dtype(), DType::Struct(..) | DType::List(..))
            });

        if needs_metadata {
            let lhs_metadata = filter_child(lhs.metadata())?;
            let rhs_metadata = filter_child(rhs.metadata())?;
            if !all_non_distinct(&lhs_metadata, &rhs_metadata, ctx)? {
                let equal = match (typed_pair, value_pair) {
                    (None, Some((lhs_val, rhs_val))) => compare_unshredded_variants(
                        &lhs_metadata,
                        &rhs_metadata,
                        &filter_child(lhs_val)?,
                        &filter_child(rhs_val)?,
                        ctx,
                    )?,
                    _ => compare_active_scalars(lhs_arr, rhs_arr, active_mask.as_ref(), ctx)?,
                };
                return Ok(Some(Scalar::bool(equal, Nullability::NonNullable)));
            }
        }

        if let Some((lhs_typed, rhs_typed)) = typed_pair {
            let lhs_typed = filter_child(lhs_typed)?;
            let rhs_typed = filter_child(rhs_typed)?;
            if !all_non_distinct(&lhs_typed, &rhs_typed, ctx)? {
                let equal = if needs_metadata {
                    compare_active_scalars(lhs_arr, rhs_arr, active_mask.as_ref(), ctx)?
                } else {
                    false
                };
                return Ok(Some(Scalar::bool(equal, Nullability::NonNullable)));
            }
        }

        if let Some((lhs_value, rhs_value)) = value_pair {
            let lhs_value = filter_child(lhs_value)?;
            let rhs_value = filter_child(rhs_value)?;
            if !all_non_distinct(&lhs_value, &rhs_value, ctx)? {
                let equal = if typed_pair.is_none() {
                    let lhs_metadata = filter_child(lhs.metadata())?;
                    let rhs_metadata = filter_child(rhs.metadata())?;
                    compare_unshredded_variants(
                        &lhs_metadata,
                        &rhs_metadata,
                        &lhs_value,
                        &rhs_value,
                        ctx,
                    )?
                } else {
                    compare_active_scalars(lhs_arr, rhs_arr, active_mask.as_ref(), ctx)?
                };
                return Ok(Some(Scalar::bool(equal, Nullability::NonNullable)));
            }
        }

        Ok(Some(Scalar::bool(true, Nullability::NonNullable)))
    }
}

/// Compare unshredded `ParquetVariant` rows by decoding raw `(metadata, value)` bytes directly
/// into `parquet_variant::Variant`, preserving Parquet Variant type tags such as `Uuid` vs
/// `String` while treating alternate binary encodings of the same logical value (e.g. short vs
/// long strings or reordered object fields) as equal.
fn compare_unshredded_variants(
    lhs_metadata: &ArrayRef,
    rhs_metadata: &ArrayRef,
    lhs_value: &ArrayRef,
    rhs_value: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let lhs_meta = lhs_metadata.clone().execute::<VarBinViewArray>(ctx)?;
    let rhs_meta = rhs_metadata.clone().execute::<VarBinViewArray>(ctx)?;
    let lhs_val = lhs_value.clone().execute::<VarBinViewArray>(ctx)?;
    let rhs_val = rhs_value.clone().execute::<VarBinViewArray>(ctx)?;

    let lhs_val_mask = lhs_val.validity()?.execute_mask(lhs_val.len(), ctx)?;
    let rhs_val_mask = rhs_val.validity()?.execute_mask(rhs_val.len(), ctx)?;

    for idx in 0..lhs_val.len() {
        let lhs_meta_bytes;
        let lhs_val_bytes;
        let lhs_variant = if lhs_val_mask.value(idx) {
            lhs_meta_bytes = lhs_meta.bytes_at(idx);
            lhs_val_bytes = lhs_val.bytes_at(idx);
            PqVariant::try_new(&lhs_meta_bytes, &lhs_val_bytes)
                .map_err(|e| vortex_err!("Invalid Parquet Variant: {e}"))?
        } else {
            PqVariant::Null
        };

        let rhs_meta_bytes;
        let rhs_val_bytes;
        let rhs_variant = if rhs_val_mask.value(idx) {
            rhs_meta_bytes = rhs_meta.bytes_at(idx);
            rhs_val_bytes = rhs_val.bytes_at(idx);
            PqVariant::try_new(&rhs_meta_bytes, &rhs_val_bytes)
                .map_err(|e| vortex_err!("Invalid Parquet Variant: {e}"))?
        } else {
            PqVariant::Null
        };

        if !pq_variants_equal(&lhs_variant, &rhs_variant) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn pq_variants_equal(lhs: &PqVariant<'_, '_>, rhs: &PqVariant<'_, '_>) -> bool {
    match (lhs, rhs) {
        (PqVariant::String(a), PqVariant::String(b)) => a == b,
        (PqVariant::String(a), PqVariant::ShortString(b))
        | (PqVariant::ShortString(b), PqVariant::String(a)) => *a == b.as_str(),
        (PqVariant::ShortString(a), PqVariant::ShortString(b)) => a.as_str() == b.as_str(),
        (PqVariant::Float(a), PqVariant::Float(b)) => a.to_bits() == b.to_bits(),
        (PqVariant::Double(a), PqVariant::Double(b)) => a.to_bits() == b.to_bits(),
        (PqVariant::List(a), PqVariant::List(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(la, rb)| pq_variants_equal(&la, &rb))
        }
        (PqVariant::Object(a), PqVariant::Object(b)) => {
            // The Parquet Variant spec requires object fields to be sorted lexicographically by
            // name, so zipping both iterators compares fields by name.
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|((lk, lv), (rk, rv))| lk == rk && pq_variants_equal(&lv, &rv))
        }
        _ => lhs == rhs,
    }
}

/// Fallback comparison for shredded or mixed-layout `ParquetVariant` rows via decoded scalars.
///
/// Note: `ParquetVariant::execute_scalar` currently decodes `PqVariant::Uuid` to `Scalar::utf8`
/// (`operations.rs`), so this path treats UUIDs as UTF-8 strings until `ParquetVariant` maps UUIDs
/// to the UUID extension type.
fn compare_active_scalars(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    active_mask: Option<&Mask>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mut lhs_probe = RepeatedArrayProbe::new(lhs.clone());
    let mut rhs_probe = RepeatedArrayProbe::new(rhs.clone());
    match active_mask {
        Some(mask) => {
            for idx in mask.to_bit_buffer().set_indices() {
                let lhs_scalar = lhs_probe.execute_scalar(idx, ctx)?;
                let rhs_scalar = rhs_probe.execute_scalar(idx, ctx)?;
                if !variant_scalars_equal(&lhs_scalar, &rhs_scalar) {
                    return Ok(false);
                }
            }
        }
        None => {
            for idx in 0..lhs.len() {
                let lhs_scalar = lhs_probe.execute_scalar(idx, ctx)?;
                let rhs_scalar = rhs_probe.execute_scalar(idx, ctx)?;
                if !variant_scalars_equal(&lhs_scalar, &rhs_scalar) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}
