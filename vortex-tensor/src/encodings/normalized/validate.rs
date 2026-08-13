// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;
use crate::utils::validate_tensor_float_input;

/// Validates the structural invariants of a [`Normalized`] array's slots.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
pub(super) fn validate_normalized_children(
    normalized: &ArrayRef,
    norms: &ArrayRef,
    validity: Option<&ArrayRef>,
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

    vortex_ensure_eq!(
        *normalized.dtype(),
        dtype.as_nonnullable(),
        "Normalized normalized child must be the non-nullable array dtype ({}), got {}",
        dtype.as_nonnullable(),
        normalized.dtype(),
    );

    let expected_norms_dtype = DType::Primitive(element_ptype, Nullability::NonNullable);
    vortex_ensure_eq!(
        *norms.dtype(),
        expected_norms_dtype,
        "Normalized norms must be a non-nullable {element_ptype} column ({expected_norms_dtype}), \
         got {}",
        norms.dtype(),
    );

    if let Some(validity) = validity {
        vortex_ensure!(
            dtype.is_nullable(),
            "Normalized must only carry a validity slot when its dtype is nullable, got {dtype}",
        );
        vortex_ensure_eq!(
            *validity.dtype(),
            Validity::DTYPE,
            "Normalized validity must be a {} column, got {}",
            Validity::DTYPE,
            validity.dtype(),
        );
        vortex_ensure_eq!(
            validity.len(),
            len,
            "Normalized validity must have the array length ({len}), got {}",
            validity.len(),
        );
    }

    Ok(())
}

/// Validates the semantic invariants documented by [`Normalized`].
///
/// The zero relationship is checked in both directions. Otherwise, a zero row with a nonzero
/// stored norm would decode differently from [`L2Norm`]. This `O(len * list_size)` scan includes
/// rows that the parent might mark null.
///
/// # Errors
///
/// Returns an error if either child has an incompatible dtype or length, or if a row violates the
/// semantic invariants.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
/// [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
pub fn validate_normalized_rows(
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
    let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;
    let norms = norms
        .map(|norms| norms.clone().execute::<PrimitiveArray>(ctx))
        .transpose()?;

    match_each_float_ptype!(element_ptype, |T| {
        let stored_norms = norms.as_ref().map(|norms| norms.as_slice::<T>());

        for i in 0..row_count {
            let (row_norm_sq, is_zero_row) =
                flat.row::<T>(i)
                    .iter()
                    .fold((0.0f64, true), |(sum_sq, is_zero), x| {
                        let value = ToPrimitive::to_f64(x).unwrap_or(f64::NAN);
                        // A valid dense f16 unit vector can have every coordinate below tolerance.
                        (sum_sq + value * value, is_zero && x.is_zero())
                    });
            let row_norm = row_norm_sq.sqrt();

            vortex_ensure!(
                row_norm.is_zero() || (row_norm - 1.0).abs() <= tolerance,
                "Normalized normalized child must have L2 norm 1.0 or 0.0, but row {i} has \
                 {row_norm:.6}",
            );

            if let Some(stored_norms) = stored_norms {
                let stored_norm_f64 = ToPrimitive::to_f64(&stored_norms[i]).unwrap_or(f64::NAN);
                vortex_ensure!(
                    stored_norm_f64 >= 0.0,
                    "Normalized norms must be non-negative, but row {i} has {stored_norm_f64:.6}",
                );

                vortex_ensure!(
                    is_zero_row == stored_norm_f64.is_zero(),
                    "Normalized normalized child must be all zeros exactly when its stored norm is \
                     0.0, but row {i} pairs a {} normalized row with a stored norm of \
                     {stored_norm_f64:.6}",
                    if is_zero_row { "zero" } else { "nonzero" },
                );
            }
        }
    });

    Ok(())
}
