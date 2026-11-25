// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::VariantArray;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Compare two decoded Variant scalars semantically: object fields are matched by name
/// regardless of storage or shredded schema order, while lists are compared positionally.
pub fn variant_scalars_equal(lhs: &Scalar, rhs: &Scalar) -> bool {
    match (lhs.is_null(), rhs.is_null()) {
        (true, true) => return true,
        (true, false) | (false, true) => return false,
        (false, false) => {}
    }

    match (lhs.dtype(), rhs.dtype()) {
        (DType::Variant(_), DType::Variant(_)) => {
            match (lhs.as_variant().value(), rhs.as_variant().value()) {
                (Some(l), Some(r)) => variant_scalars_equal(l, r),
                (None, None) => true,
                _ => false,
            }
        }
        (DType::Struct(..), DType::Struct(..)) => {
            let lhs_struct = lhs.as_struct();
            let rhs_struct = rhs.as_struct();
            let lhs_names = lhs_struct.names();
            if lhs_names.len() != rhs_struct.names().len() {
                return false;
            }
            let Some(lhs_fields) = lhs_struct.fields_iter() else {
                return rhs_struct.is_null();
            };
            lhs_names.iter().zip(lhs_fields).all(|(name, lhs_field)| {
                rhs_struct
                    .field(name.as_ref())
                    .is_some_and(|rhs_field| variant_scalars_equal(&lhs_field, &rhs_field))
            })
        }
        (
            DType::List(..) | DType::FixedSizeList(..),
            DType::List(..) | DType::FixedSizeList(..),
        ) => {
            let lhs_list = lhs.as_list();
            let rhs_list = rhs.as_list();
            if lhs_list.len() != rhs_list.len() {
                return false;
            }
            (0..lhs_list.len()).all(|idx| {
                matches!(
                    (lhs_list.element(idx), rhs_list.element(idx)),
                    (Some(l), Some(r)) if variant_scalars_equal(&l, &r)
                )
            })
        }
        _ => lhs == rhs,
    }
}

/// Checks whether two canonical variant arrays are element-wise non-distinct.
///
/// Variant values cannot be routed back through [`all_non_distinct`]: canonicalizing a variant
/// value array yields another canonical variant (with no shredded tree), which would recurse
/// forever. The generic fallback therefore compares logical variant scalars row-by-row. Encodings
/// that can compare their typed/value children more cheaply (e.g. `ParquetVariant`) register an
/// aggregate kernel that intercepts the comparison before it reaches this fallback.
///
/// [`all_non_distinct`]: super::all_non_distinct
pub(super) fn check_variant_identical(
    lhs: &VariantArray,
    rhs: &VariantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if lhs.len() != rhs.len() {
        return Ok(false);
    }
    for idx in 0..lhs.len() {
        let lhs_scalar = lhs.execute_scalar(idx, ctx)?;
        let rhs_scalar = rhs.execute_scalar(idx, ctx)?;
        if !variant_scalars_equal(&lhs_scalar, &rhs_scalar) {
            return Ok(false);
        }
    }
    Ok(true)
}
