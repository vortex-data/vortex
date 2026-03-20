// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Float;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::ToCanonical;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::ScalarFnArray;
use vortex::array::match_each_float_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::extension::ExtDType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
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
#[derive(Debug, Clone)]
pub struct NormVectorArray {
    /// The backing vector array that has been unit normalized.
    ///
    /// The underlying elements of the vector array must be floating-point.
    pub(crate) vector_array: ArrayRef,

    /// The L2 (Frobenius) norms of each vector.
    ///
    /// This must have the same dtype as the elements of the vector array.
    pub(crate) norms: ArrayRef,
}

impl NormVectorArray {
    /// Creates a new [`NormVectorArray`] from a unit-normalized vector array and its L2 norms.
    ///
    /// The `vector_array` must be a [`Vector`] extension array with floating-point elements, and
    /// `norms` must be a primitive array of the same float type with the same length.
    pub fn try_new(vector_array: ArrayRef, norms: ArrayRef) -> VortexResult<Self> {
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

        let element_ptype = extension_element_ptype(ext)?;

        let expected_norms_dtype = DType::Primitive(element_ptype, Nullability::NonNullable);
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
        })
    }

    /// Encodes a [`Vector`] extension array into a [`NormVectorArray`] by computing L2 norms and
    /// dividing each vector by its norm.
    ///
    /// The input must be a [`Vector`] extension array with floating-point elements.
    pub fn compress(vector_array: ArrayRef) -> VortexResult<Self> {
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

        let list_size = extension_list_size(ext)?;
        let row_count = vector_array.len();

        // Compute L2 norms using the scalar function.
        let l2_norm_fn = ScalarFn::new(L2Norm, EmptyOptions).erased();
        let norms = ScalarFnArray::try_new(l2_norm_fn, vec![vector_array.clone()], row_count)?
            .to_primitive()
            .into_array();

        // Divide each vector element by its corresponding norm.
        let storage = extension_storage(&vector_array)?;
        let flat = extract_flat_elements(&storage, list_size)?;
        let norms_prim = norms.to_canonical()?.into_primitive();

        match_each_float_ptype!(flat.ptype(), |T| {
            let norms_slice = norms_prim.as_slice::<T>();

            let normalized_elems: PrimitiveArray = (0..row_count)
                .flat_map(|i| {
                    let inv_norm = safe_inv_norm(norms_slice[i]);
                    flat.row::<T>(i).iter().map(move |&v| v * inv_norm)
                })
                .collect();

            let fsl = FixedSizeListArray::new(
                normalized_elems.into_array(),
                u32::try_from(list_size)?,
                Validity::NonNullable,
                row_count,
            );

            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            let normalized_vector = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();

            Self::try_new(normalized_vector, norms)
        })
    }

    /// Returns a reference to the backing vector array that has been unit normalized.
    pub fn vector_array(&self) -> &ArrayRef {
        &self.vector_array
    }

    /// Returns a reference to the L2 (Frobenius) norms of each vector.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }

    /// Reconstructs the original vectors by multiplying each unit-normalized vector by its L2 norm.
    pub fn decompress(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ext_dtype = self
            .vector_array
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "expected Vector extension dtype, got {}",
                    self.vector_array.dtype()
                )
            })?;

        let list_size = extension_list_size(ext_dtype)?;
        let row_count = self.vector_array.len();

        let storage = extension_storage(&self.vector_array)?;
        let flat = extract_flat_elements(&storage, list_size)?;

        let norms_prim = self.norms.to_canonical()?.into_primitive();

        match_each_float_ptype!(flat.ptype(), |T| {
            let norms_slice = norms_prim.as_slice::<T>();

            let result_elems: PrimitiveArray = (0..row_count)
                .flat_map(|i| {
                    let norm = norms_slice[i];
                    flat.row::<T>(i).iter().map(move |&v| v * norm)
                })
                .collect();

            let fsl = FixedSizeListArray::new(
                result_elems.into_array(),
                u32::try_from(list_size)?,
                Validity::NonNullable,
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
