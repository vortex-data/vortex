// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;

use crate::utils::extension_element_ptype;
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

    /// Returns a reference to the backing vector array that has been unit normalized.
    pub fn vector_array(&self) -> &ArrayRef {
        &self.vector_array
    }

    /// Returns a reference to the L2 (Frobenius) norms of each vector.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }

    // TODO docs
    pub(super) fn execute_into_vector(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        todo!()
    }
}
