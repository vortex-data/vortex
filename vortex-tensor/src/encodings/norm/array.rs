// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Float;
use num_traits::Zero;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_float_ptype;
use vortex::array::stats::ArrayStats;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::extension::ExtDType;
use vortex::dtype::extension::ExtDTypeRef;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::extension::EmptyMetadata;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFn;

use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::extension_element_ptype;
use crate::utils::extension_list_size;
use crate::utils::extension_storage;
use crate::utils::extract_flat_elements;
use crate::vector::Vector;

/// A normalized array that stores unit-normalized vectors alongside their original L2 norms.
///
/// Each vector in the array is divided by its L2 norm, producing a unit-normalized vector. The
/// original norms are stored separately so that the original vectors can be reconstructed.
///
/// The `vector_array` child carries its own validity and nullability, so a nullable input vector
/// array produces a nullable `NormVectorArray`.
#[derive(Debug, Clone)]
pub struct NormVectorArray {
    /// The backing vector array that has been unit normalized.
    ///
    /// The underlying elements of the vector array must be floating-point. This child may be
    /// nullable; its validity determines the validity of the `NormVectorArray`.
    pub(crate) vector_array: ArrayRef,

    /// The L2 norms of each vector.
    ///
    /// This must have the same dtype as the elements of the vector array.
    pub(crate) norms: ArrayRef,

    /// Stats set owned by this array.
    pub(crate) stats_set: ArrayStats,
}

impl NormVectorArray {
    /// Creates a new [`NormVectorArray`] from a unit-normalized vector array and associated L2
    /// norms for each vector.
    ///
    /// The `vector_array` must be a [`Vector`] extension array with floating-point elements, and
    /// `norms` must be a primitive array of the same float type with the same length. The
    /// `vector_array` may be nullable.
    pub fn try_new(vector_array: ArrayRef, norms: ArrayRef) -> VortexResult<Self> {
        let ext = Self::validate(&vector_array)?;

        let element_ptype = extension_element_ptype(&ext)?;

        let nullability = Nullability::from(vector_array.dtype().is_nullable());
        let expected_norms_dtype = DType::Primitive(element_ptype, nullability);
        vortex_ensure_eq!(
            *norms.dtype(),
            expected_norms_dtype,
            "norms dtype must match vector element type"
        );

        vortex_ensure_eq!(
            vector_array.len(),
            norms.len(),
            "vector_array and norms must have the same length"
        );

        Ok(Self {
            vector_array,
            norms,
            stats_set: ArrayStats::default(),
        })
    }

    /// Validates that the given array has the [`Vector`] extension type and returns the extension
    /// dtype.
    fn validate(vector_array: &ArrayRef) -> VortexResult<ExtDTypeRef> {
        let ext = vector_array.dtype().as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "vector_array dtype must be an extension type, got {}",
                vector_array.dtype()
            )
        })?;

        vortex_ensure!(
            ext.is::<Vector>(),
            "vector_array must have the Vector extension type, got {}",
            vector_array.dtype()
        );

        Ok(ext.clone())
    }

    /// Encodes a [`Vector`] extension array into a [`NormVectorArray`] by computing L2 norms and
    /// dividing each vector by its norm.
    ///
    /// The input must be a [`Vector`] extension array with floating-point elements. Nullable inputs
    /// are supported; the validity mask is preserved and the normalized data for null rows is
    /// unspecified.
    ///
    /// Note that compression is lossy per floating-point operations.
    pub fn compress(vector_array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let ext = Self::validate(&vector_array)?;

        let list_size = extension_list_size(&ext)?;
        let row_count = vector_array.len();
        let nullability = Nullability::from(vector_array.dtype().is_nullable());
        let validity = vector_array.validity()?;

        // Compute L2 norms using the scalar function. If the input is nullable, the norms will
        // also be nullable (null vectors produce null norms).
        let storage = extension_storage(&vector_array)?;
        let l2_norm_expr =
            Expression::try_new(ScalarFn::new(L2Norm, EmptyOptions).erased(), [root()])?;
        let norms_prim: PrimitiveArray = vector_array.apply(&l2_norm_expr)?.execute(ctx)?;
        let norms_array = norms_prim.clone().into_array();

        // Extract flat elements from the (always non-nullable) storage for normalization.
        let flat = extract_flat_elements(&storage, list_size, ctx)?;

        match_each_float_ptype!(flat.ptype(), |T| {
            let norms_slice = norms_prim.as_slice::<T>();

            let normalized_elems: PrimitiveArray = (0..row_count)
                .map(|i| -> VortexResult<Vec<T>> {
                    if !validity.is_valid(i)? {
                        return Ok(vec![T::zero(); list_size]);
                    }

                    let inv_norm = safe_inv_norm(norms_slice[i]);
                    Ok(flat.row::<T>(i).iter().map(|&v| v * inv_norm).collect())
                })
                .collect::<VortexResult<Vec<Vec<T>>>>()?
                .into_iter()
                .flatten()
                .collect();

            // Reconstruct the vector array with the same nullability as the input.
            let validity = Validity::from(nullability);
            let fsl = FixedSizeListArray::new(
                normalized_elems.into_array(),
                u32::try_from(list_size)?,
                validity,
                row_count,
            );

            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            let normalized_vector = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();

            Self::try_new(normalized_vector, norms_array)
        })
    }

    /// Returns a reference to the backing vector array that has been unit normalized.
    pub fn vector_array(&self) -> &ArrayRef {
        &self.vector_array
    }

    /// Returns a reference to the L2 norms of each vector.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }

    /// Reconstructs the original vectors by multiplying each unit-normalized vector by its L2 norm.
    ///
    /// The returned array has the same dtype (including nullability) as the original
    /// `vector_array` child.
    pub fn decompress(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ext = Self::validate(&self.vector_array)?;
        let nullability = Nullability::from(self.vector_array.dtype().is_nullable());

        let list_size = extension_list_size(&ext)?;
        let row_count = self.vector_array.len();

        let storage = extension_storage(&self.vector_array)?;
        let flat = extract_flat_elements(&storage, list_size, ctx)?;

        let norms_prim: PrimitiveArray = self.norms.clone().execute(ctx)?;

        match_each_float_ptype!(flat.ptype(), |T| {
            let norms_slice = norms_prim.as_slice::<T>();

            let result_elems: PrimitiveArray = (0..row_count)
                .flat_map(|i| {
                    let norm = norms_slice[i];
                    flat.row::<T>(i).iter().map(move |&v| v * norm)
                })
                .collect();

            let validity = Validity::from(nullability);
            let fsl = FixedSizeListArray::new(
                result_elems.into_array(),
                u32::try_from(list_size)?,
                validity,
                row_count,
            );

            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
        })
    }
}

/// Returns `1 / norm` if the norm is non-zero, or zero otherwise.
///
/// This avoids division by zero for zero-length or all-zero vectors.
fn safe_inv_norm<T: Float>(norm: T) -> T {
    if norm == T::zero() {
        T::zero()
    } else {
        T::one() / norm
    }
}
