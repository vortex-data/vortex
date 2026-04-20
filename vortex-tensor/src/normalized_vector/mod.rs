// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Normalized vector extension type: a refinement of [`Vector`](crate::vector::Vector) whose
//! rows are guaranteed (or asserted, for lossy encodings) to have unit L2 norm.

use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::PType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// Refinement of [`Vector`](crate::vector::Vector) that asserts every valid row is L2-normalized
/// (unit-norm) or the zero vector.
///
/// The storage shape is identical to [`Vector`](crate::vector::Vector): a `FixedSizeList<float,
/// dim, nullability>` with non-nullable float elements. Downstream operators such as
/// [`L2Denorm`](crate::scalar_fns::l2_denorm::L2Denorm),
/// [`L2Norm`](crate::scalar_fns::l2_norm::L2Norm),
/// [`InnerProduct`](crate::scalar_fns::inner_product::InnerProduct), and
/// [`CosineSimilarity`](crate::scalar_fns::cosine_similarity::CosineSimilarity) short-circuit
/// arithmetic when they see this refinement.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NormalizedVector;

impl NormalizedVector {
    /// Wraps `storage` as a [`NormalizedVector`] extension array after checking that every valid
    /// row is unit-norm or the zero vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the extension dtype rejects `storage`, if `storage` is not a tensor
    /// with float elements, or if any valid row's L2 norm is not `1.0` (or `0.0`) within the
    /// tolerance implied by the element precision.
    pub fn try_new(storage: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ext = ExtensionArray::try_new_from_vtable(NormalizedVector, EmptyMetadata, storage)?
            .into_array();
        validate_unit_norm_rows(&ext, ctx)?;
        Ok(ext)
    }

    /// Wraps `storage` as a [`NormalizedVector`] extension array **without** validating that
    /// rows are unit-norm.
    ///
    /// # Safety
    ///
    /// Every valid row must be unit-norm or the zero vector. Lossy approximations (e.g.
    /// TurboQuant) deliberately relax this, but still treat the claim as authoritative
    /// downstream. Violating this does not cause memory unsafety but will produce silently
    /// incorrect results.
    ///
    /// # Errors
    ///
    /// Returns an error if the extension dtype rejects `storage` (e.g. non-FSL storage, wrong
    /// element dtype, or nullable elements).
    pub unsafe fn new_unchecked(storage: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::try_new_from_vtable(NormalizedVector, EmptyMetadata, storage)?
                .into_array(),
        )
    }
}

/// Returns the acceptable unit-norm drift for the given element precision.
pub(crate) fn unit_norm_tolerance(element_ptype: PType) -> f64 {
    match element_ptype {
        PType::F16 => 2e-3,
        PType::F32 => 2e-6,
        PType::F64 => 1e-10,
        _ => unreachable!("NormalizedVector requires float elements, got {element_ptype:?}"),
    }
}

/// Validates that every valid row of a [`NormalizedVector`] extension array has L2 norm `1.0`
/// or `0.0` within the element-precision tolerance.
fn validate_unit_norm_rows(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let row_count = array.len();
    if row_count == 0 {
        return Ok(());
    }

    let tensor_match = validate_tensor_float_input(array.dtype())?;
    let element_ptype = tensor_match.element_ptype();
    let tolerance = unit_norm_tolerance(element_ptype);
    let tensor_flat_size = tensor_match.list_size() as usize;

    let ext: ExtensionArray = array.clone().execute(ctx)?;
    let validity = ext.as_ref().validity()?;
    let flat = extract_flat_elements(ext.storage_array(), tensor_flat_size, ctx)?;

    match_each_float_ptype!(element_ptype, |T| {
        for i in 0..row_count {
            if !validity.is_valid(i)? {
                continue;
            }

            let row_norm_sq = flat.row::<T>(i).iter().fold(0.0f64, |sum_sq, x| {
                let value = ToPrimitive::to_f64(x).unwrap_or(f64::NAN);
                sum_sq + value * value
            });
            let row_norm = row_norm_sq.sqrt();

            vortex_ensure!(
                row_norm == 0.0 || (row_norm - 1.0).abs() <= tolerance,
                "NormalizedVector row {i} has L2 norm {row_norm:.6}, expected 1.0 or 0.0",
            );
        }
    });

    Ok(())
}

mod matcher;
mod vtable;

pub use matcher::AnyNormalizedVector;
