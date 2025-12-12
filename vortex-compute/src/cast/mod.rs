// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless casting of Vortex vectors and scalars for different logical data types.

mod binaryview;
mod bool;
mod decimal;
mod dvector;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod pvector;
mod struct_;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::match_each_scalar;
use vortex_vector::match_each_vector;
use vortex_vector::null::NullScalar;
use vortex_vector::null::NullVector;

/// Trait for casting vectors and scalars to different data types.
///
/// # Nullability Requirements
///
/// Casting a source that contains null values to a non-nullable target dtype will return an error.
/// This invariant is not checked by the common helper functions, so each implementation is
/// responsible for enforcing it.
///
/// # Common Casting Behaviors
///
/// All implementations share these behaviors:
/// - **Identity**: Casting to the same dtype (with compatible nullability) returns a clone.
/// - **Null casting**: Any all-null vector can be cast to [`DType::Null`], and any null scalar
///   can be cast to a [`NullScalar`].
/// - **Extension types**: Casting to an extension type delegates to casting to its storage dtype.
pub trait Cast {
    /// The output type after casting.
    type Output;

    /// Cast the vector or scalar to the specified data type.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Self::Output>;
}

impl Cast for Datum {
    type Output = Datum;

    /// Dispatches to the contained [`Scalar`] or [`Vector`] cast implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Datum> {
        Ok(match self {
            Datum::Scalar(scalar) => scalar.cast(target_dtype)?.into(),
            Datum::Vector(vector) => vector.cast(target_dtype)?.into(),
        })
    }
}

impl Cast for Scalar {
    type Output = Scalar;

    /// Dispatches to the underlying typed scalar implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        match_each_scalar!(self, |s| { Cast::cast(s, target_dtype) })
    }
}

impl Cast for Vector {
    type Output = Vector;

    /// Dispatches to the underlying typed vector implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        match_each_vector!(self, |v| { Cast::cast(v, target_dtype) })
    }
}

/// Handles common vector cast cases: Null (if all-null) and Extension (delegate to storage).
///
/// Returns `Ok(Some(...))` if handled, `Err(...)` on error, or `Ok(None)` to fall through.
pub(crate) fn try_cast_vector_common<V: Cast<Output = Vector> + VectorOps>(
    vector: &V,
    target_dtype: &DType,
) -> VortexResult<Option<Vector>> {
    match target_dtype {
        DType::Null if vector.validity().all_false() => {
            Ok(Some(NullVector::new(vector.len()).into()))
        }
        DType::Extension(ext_dtype) => vector.cast(ext_dtype.storage_dtype()).map(Some),
        _ => Ok(None),
    }
}

/// Handles common scalar cast cases: Null (if null) and Extension (delegate to storage).
///
/// Returns `Ok(Some(...))` if handled, `Err(...)` on error, or `Ok(None)` to fall through.
pub(crate) fn try_cast_scalar_common<S: Cast<Output = Scalar> + ScalarOps>(
    scalar: &S,
    target_dtype: &DType,
) -> VortexResult<Option<Scalar>> {
    match target_dtype {
        DType::Null if !scalar.is_valid() => Ok(Some(NullScalar.into())),
        DType::Extension(ext_dtype) => scalar.cast(ext_dtype.storage_dtype()).map(Some),
        _ => Ok(None),
    }
}
