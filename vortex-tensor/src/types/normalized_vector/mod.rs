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
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::types::vector::AnyVector;
use crate::types::vector::Vector;
use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;

/// Refinement of [`Vector`](crate::vector::Vector) that asserts every valid row is L2-normalized
/// (unit-norm) or the zero vector.
///
/// The storage dtype is `DType::Extension(Vector(FixedSizeList<float, dim>))`, i.e. a
/// [`Vector`](crate::vector::Vector) extension array. Downstream operators such as
/// [`L2Denorm`](crate::scalar_fns::l2_denorm::L2Denorm),
/// [`L2Norm`](crate::scalar_fns::l2_norm::L2Norm),
/// [`InnerProduct`](crate::scalar_fns::inner_product::InnerProduct), and
/// [`CosineSimilarity`](crate::scalar_fns::cosine_similarity::CosineSimilarity) short-circuit
/// arithmetic when they see this refinement.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NormalizedVector;

impl NormalizedVector {
    /// Wraps a [`FixedSizeList`](vortex_array::arrays::FixedSizeListArray) of float elements
    /// as a [`NormalizedVector`] extension array, wrapping the FSL in a
    /// [`Vector`](crate::vector::Vector) first.
    ///
    /// Every valid row is checked to be unit-norm or the zero vector before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if `fsl` is not a `FixedSizeList` of non-nullable float elements, or if
    /// any valid row's L2 norm is not `1.0` (or `0.0`) within the tolerance implied by the
    /// element precision.
    pub fn try_new(fsl: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let vector = Vector::try_new_vector_array(fsl)?;
        // Validate before wrapping so we iterate the inner `Vector` storage directly. The
        // `ExtensionArray::try_new_from_vtable` call below runs `validate_dtype` (which only
        // checks the storage dtype shape), but the unit-norm check is a bulk row operation we
        // run explicitly here.
        validate_unit_norm_rows(&vector, ctx)?;
        Ok(
            ExtensionArray::try_new_from_vtable(NormalizedVector, EmptyMetadata, vector)?
                .into_array(),
        )
    }

    /// Wraps `fsl` as a [`NormalizedVector`] extension array **without** validating that rows
    /// are unit-norm. The FSL is still wrapped in a [`Vector`](crate::vector::Vector) first.
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
    /// Returns an error if `fsl` is not a `FixedSizeList` of non-nullable float elements.
    pub unsafe fn new_unchecked(fsl: ArrayRef) -> VortexResult<ArrayRef> {
        let vector = Vector::try_new_vector_array(fsl)?;
        Ok(
            ExtensionArray::try_new_from_vtable(NormalizedVector, EmptyMetadata, vector)?
                .into_array(),
        )
    }

    /// Wraps an already-constructed [`Vector`](crate::vector::Vector) extension array as a
    /// [`NormalizedVector`] **without** validating that rows are unit-norm.
    ///
    /// # Safety
    ///
    /// Every valid row of `vector` must be unit-norm or the zero vector.
    ///
    /// # Errors
    ///
    /// Returns an error if `vector.dtype()` is not a `Vector` extension dtype.
    pub unsafe fn wrap_vector_unchecked(vector: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::try_new_from_vtable(NormalizedVector, EmptyMetadata, vector)?
                .into_array(),
        )
    }
}

/// Validates that every valid row of a [`Vector`](crate::vector::Vector) extension array has L2
/// norm `1.0` or `0.0` within the element-precision tolerance.
///
/// The input is expected to be a `Vector` extension array (not a raw `FixedSizeList`), matching
/// the storage of a `NormalizedVector`.
pub(crate) fn validate_unit_norm_rows(
    vector_array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let row_count = vector_array.len();
    if row_count == 0 {
        return Ok(());
    }

    let vector_metadata = vector_array.dtype().as_extension().metadata::<AnyVector>();
    let element_ptype = vector_metadata.element_ptype();
    let dim = vector_metadata.dimensions() as usize;
    let tolerance = unit_norm_tolerance(element_ptype, dim);

    let ext: ExtensionArray = vector_array.clone().execute(ctx)?;
    let validity = ext.as_ref().validity()?;
    let flat = extract_flat_elements(ext.storage_array(), dim, ctx)?;

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

/// Returns the underlying `FixedSizeList` storage dtype for a vector-shaped extension dtype.
///
/// For a plain [`Vector`], this is the direct storage dtype. For a [`NormalizedVector`]
/// refinement it drills through one extra extension layer.
pub(crate) fn vector_fsl_storage_dtype(
    ext: &vortex_array::dtype::extension::ExtDTypeRef,
) -> Option<vortex_array::dtype::DType> {
    use vortex_array::dtype::DType;
    if ext.is::<Vector>() {
        Some(ext.storage_dtype().clone())
    } else if ext.is::<NormalizedVector>() {
        let DType::Extension(inner) = ext.storage_dtype() else {
            return None;
        };
        if !inner.is::<Vector>() {
            return None;
        }
        Some(inner.storage_dtype().clone())
    } else {
        None
    }
}

/// Returns the underlying `Vector` extension array inside a vector-shaped extension array.
///
/// For a [`NormalizedVector`] array, this executes the outer extension and returns its
/// `Vector` storage child. For a plain [`Vector`] array, it returns the array itself (after
/// canonicalizing to an `ExtensionArray`).
pub(crate) fn inner_vector_array(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let is_normalized = array
        .dtype()
        .as_extension_opt()
        .is_some_and(|ext| ext.is::<NormalizedVector>());
    if is_normalized {
        let ext: ExtensionArray = array.clone().execute(ctx)?;
        Ok(ext.storage_array().clone())
    } else {
        Ok(array.clone())
    }
}

mod matcher;
mod vtable;

pub use matcher::AnyNormalizedVector;
