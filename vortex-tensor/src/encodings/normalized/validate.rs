// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_float_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;
use crate::utils::validate_tensor_float_input;

/// Validates the structural invariants of a [`Normalized`] array's children.
///
/// These are the cheap, dtype-and-length checks that every [`NormalizedArray`] upholds, whichever
/// constructor built it. They run on construction and on deserialization.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
/// [`NormalizedArray`]: crate::encodings::normalized::NormalizedArray
pub(super) fn validate_normalized_children(
    normalized: &ArrayRef,
    norms: &ArrayRef,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        normalized.len(),
        len,
        "Normalized normalized child must have the array length ({len}), got {}",
        normalized.len(),
    );
    vortex_ensure_eq!(
        norms.len(),
        len,
        "Normalized norms child must have the array length ({len}), got {}",
        norms.len(),
    );

    let tensor_match = validate_tensor_float_input(normalized.dtype())?;
    let element_ptype = tensor_match.element_ptype();

    let DType::Primitive(norms_ptype, _) = norms.dtype() else {
        vortex_bail!(
            "Normalized norms must be a primitive float array, got {}",
            norms.dtype(),
        );
    };
    vortex_ensure_eq!(
        *norms_ptype,
        element_ptype,
        "Normalized norms dtype must match the normalized element dtype ({element_ptype}), \
         got {norms_ptype}",
    );

    let expected = normalized
        .dtype()
        .union_nullability(norms.dtype().nullability());
    vortex_ensure_eq!(
        *dtype,
        expected,
        "Normalized dtype must be the union of its children's nullability ({expected}), got {dtype}",
    );

    Ok(())
}

/// Validates that `normalized` and (when supplied) the matching `norms` jointly satisfy the
/// semantic [`Normalized`] invariants:
///
/// - Every valid row of `normalized` has L2 norm `1.0` or `0.0`, within the tolerance implied by
///   the element precision.
/// - When `norms` is supplied, every stored norm is non-negative and any row whose stored norm is
///   `0.0` is exactly the zero vector in `normalized`.
///
/// This costs `O(len * list_size)`, which is why it is a separate step rather than part of the
/// encoding's structural validation.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
pub fn validate_l2_normalized_rows_against_norms(
    normalized: &ArrayRef,
    norms: Option<&ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let row_count = normalized.len();
    if row_count == 0 {
        return Ok(());
    }

    let tensor_match = validate_tensor_float_input(normalized.dtype())?;
    let element_ptype = tensor_match.element_ptype();
    let tensor_flat_size = tensor_match.list_size() as usize;
    let tolerance = unit_norm_tolerance(element_ptype, tensor_flat_size);

    if let Some(norms) = norms {
        vortex_ensure_eq!(
            norms.len(),
            row_count,
            "Normalized norms must have the same length as the normalized child ({row_count}), \
             got {}",
            norms.len(),
        );

        let DType::Primitive(norms_ptype, _) = norms.dtype() else {
            vortex_bail!(
                "Normalized norms must be a primitive float array, got {}",
                norms.dtype(),
            );
        };
        vortex_ensure_eq!(
            *norms_ptype,
            element_ptype,
            "Normalized norms ptype must match the normalized element ptype ({element_ptype}), \
             got {norms_ptype}",
        );
    }

    let normalized: ExtensionArray = normalized.clone().execute(ctx)?;
    let normalized_validity = normalized.as_ref().validity()?;

    let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;
    let norms = norms
        .map(|norms| norms.clone().execute::<PrimitiveArray>(ctx))
        .transpose()?;

    let combined_validity = match &norms {
        Some(norms) => normalized_validity.and(norms.validity()?)?,
        None => normalized_validity,
    };

    // Resolve validity to a mask once rather than probing it per row.
    let combined_valid = combined_validity.execute_mask(row_count, ctx)?;

    match_each_float_ptype!(element_ptype, |T| {
        let stored_norms = norms.as_ref().map(|norms| norms.as_slice::<T>());

        for i in 0..row_count {
            if !combined_valid.value(i) {
                continue;
            }

            let (row_norm_sq, is_zero_row) =
                flat.row::<T>(i)
                    .iter()
                    .fold((0.0f64, true), |(sum_sq, is_zero), x| {
                        let value = ToPrimitive::to_f64(x).unwrap_or(f64::NAN);
                        (sum_sq + value * value, is_zero && value.abs() <= tolerance)
                    });
            let row_norm = row_norm_sq.sqrt();

            vortex_ensure!(
                row_norm == 0.0 || (row_norm - 1.0).abs() <= tolerance,
                "Normalized normalized child must have L2 norm 1.0 or 0.0, but row {i} has \
                 {row_norm:.6}",
            );

            if let Some(stored_norms) = stored_norms {
                let stored_norm_f64 = ToPrimitive::to_f64(&stored_norms[i]).unwrap_or(f64::NAN);
                vortex_ensure!(
                    stored_norm_f64 >= 0.0,
                    "Normalized norms must be non-negative, but row {i} has {stored_norm_f64:.6}",
                );

                if stored_norm_f64 == 0.0 {
                    vortex_ensure!(
                        is_zero_row,
                        "Normalized normalized child must be all zeros when norms row {i} is 0.0",
                    );
                }
            }
        }
    });

    Ok(())
}
