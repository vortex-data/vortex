// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod filter;
mod fixed_size_list;
mod list;
mod primitive;
mod struct_;
#[cfg(test)]
mod tests;
mod varbin;
mod variant;

use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use self::bool::check_bool_identical;
use self::decimal::check_decimal_identical;
use self::extension::check_extension_identical;
use self::filter::shared_validity_mask;
use self::fixed_size_list::check_fixed_size_list_identical;
use self::list::check_list_identical;
use self::primitive::check_primitive_identical;
use self::struct_::check_struct_identical;
use self::varbin::check_varbinview_identical;
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
use crate::aggregate_fn::fns::all_non_distinct::variant::check_variant_identical;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Check if two arrays are element-wise non-distinct, treating null == null as true.
///
/// Returns `true` if and only if:
/// - Both arrays have the same dtype and length
/// - At every position, both are null or both are non-null with the same value
/// - The arrays are empty, vacuously
///
/// This is a fused `bool_all(non_distinct(lhs, rhs))` aggregate that allows early
/// termination via accumulator saturation as soon as a mismatch is found.
pub fn all_non_distinct(a: &ArrayRef, b: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    if a.dtype() != b.dtype() {
        vortex_bail!(
            "all_non_distinct: dtype mismatch: {} vs {}",
            a.dtype(),
            b.dtype()
        );
    }

    if a.len() != b.len() {
        vortex_bail!(
            "all_non_distinct: length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
    }

    if a.is_empty() {
        return Ok(true);
    }

    let Some(shared_validity) = shared_validity_mask(a, b, ctx)? else {
        return Ok(false);
    };
    if shared_validity.true_count() == 0 {
        return Ok(true);
    }

    let validity = Validity::from_mask(shared_validity, a.dtype().nullability());
    let batch = StructArray::try_new(NAMES.clone(), vec![a.clone(), b.clone()], a.len(), validity)?
        .into_array();

    let mut acc = Accumulator::try_new(AllNonDistinct, EmptyOptions, batch.dtype().clone())?;
    acc.accumulate(&batch, ctx)?;
    let result = acc.finish()?;

    Ok(result.as_bool().value().unwrap_or(false))
}

static NAMES: LazyLock<FieldNames> = LazyLock::new(|| FieldNames::from(["lhs", "rhs"]));

/// Fused `bool_all(non_distinct(lhs, rhs))` aggregate function.
///
/// This combines a pairwise non-distinct scalar comparison with a boolean-all reduction
/// into a single aggregate, enabling early termination via accumulator saturation: as soon
/// as the first distinct pair is found, the accumulator is saturated and remaining batches
/// are skipped.
///
/// Like other `all` aggregates, this is vacuously true for empty input.
///
/// The input is a `Struct{lhs: T, rhs: T}` and the result is `Bool(NonNullable)`.
#[derive(Clone, Debug)]
pub struct AllNonDistinct;

/// Partial accumulator state: just a bool tracking "all non-distinct so far".
pub struct AllNonDistinctPartial {
    all_non_distinct: bool,
}

impl AggregateFnVTable for AllNonDistinct {
    type Options = EmptyOptions;
    type Partial = AllNonDistinctPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.all_non_distinct")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("AllNonDistinct is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Struct(fields, _) if fields.nfields() == 2 => {
                let lhs = fields.fields().next()?;
                let rhs = fields.fields().nth(1)?;
                (lhs == rhs).then(|| DType::Bool(Nullability::NonNullable))
            }
            _ => None,
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(AllNonDistinctPartial {
            all_non_distinct: true,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if !partial.all_non_distinct {
            return Ok(());
        }

        if !other.as_bool().value().unwrap_or(false) {
            partial.all_non_distinct = false;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            partial.all_non_distinct,
            Nullability::NonNullable,
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.all_non_distinct = true;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        !partial.all_non_distinct
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !partial.all_non_distinct {
            return Ok(());
        }

        match batch {
            Columnar::Constant(c) => {
                let _ = c;
                Ok(())
            }
            Columnar::Canonical(c) => {
                let Canonical::Struct(s) = c else {
                    vortex_bail!(
                        "AllNonDistinct expects a Struct canonical, got {:?}",
                        c.dtype()
                    );
                };

                // The struct-level validity represents the shared validity mask
                // (positions where both lhs and rhs are non-null).
                let struct_mask = s.validity()?.execute_mask(s.len(), ctx)?;
                if struct_mask.true_count() == 0 {
                    return Ok(());
                }

                let lhs = s.unmasked_field(0);
                let rhs = s.unmasked_field(1);

                // Filter to only valid rows if the struct has nulls.
                let (lhs, rhs) = if struct_mask.true_count() == s.len() {
                    (lhs.clone(), rhs.clone())
                } else {
                    (lhs.filter(struct_mask.clone())?, rhs.filter(struct_mask)?)
                };

                let lhs_canonical = lhs.execute::<Canonical>(ctx)?;
                let rhs_canonical = rhs.execute::<Canonical>(ctx)?;

                partial.all_non_distinct =
                    check_canonical_identical(&lhs_canonical, &rhs_canonical, ctx)?;

                Ok(())
            }
        }
    }

    fn finalize(&self, _partials: ArrayRef) -> VortexResult<ArrayRef> {
        vortex_bail!("AllNonDistinct does not support array finalization");
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            partial.all_non_distinct,
            Nullability::NonNullable,
        ))
    }
}

fn check_canonical_identical(
    lhs: &Canonical,
    rhs: &Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match (lhs, rhs) {
        (Canonical::Null(_), Canonical::Null(_)) => Ok(true),
        (Canonical::Bool(lhs), Canonical::Bool(rhs)) => check_bool_identical(lhs, rhs),
        (Canonical::Primitive(lhs), Canonical::Primitive(rhs)) => {
            check_primitive_identical(lhs, rhs)
        }
        (Canonical::Decimal(lhs), Canonical::Decimal(rhs)) => check_decimal_identical(lhs, rhs),
        (Canonical::VarBinView(lhs), Canonical::VarBinView(rhs)) => {
            check_varbinview_identical(lhs, rhs)
        }
        (Canonical::Struct(lhs), Canonical::Struct(rhs)) => check_struct_identical(lhs, rhs, ctx),
        (Canonical::List(lhs), Canonical::List(rhs)) => check_list_identical(lhs, rhs, ctx),
        (Canonical::FixedSizeList(lhs), Canonical::FixedSizeList(rhs)) => {
            check_fixed_size_list_identical(lhs, rhs, ctx)
        }
        (Canonical::Extension(lhs), Canonical::Extension(rhs)) => {
            check_extension_identical(lhs, rhs, ctx)
        }
        (Canonical::Variant(lhs), Canonical::Variant(rhs)) => {
            check_variant_identical(lhs, rhs, ctx)
        }
        _ => Err(vortex_err!(
            "Canonical type mismatch in AllNonDistinct: {:?} vs {:?}",
            lhs.dtype(),
            rhs.dtype()
        )),
    }
}
