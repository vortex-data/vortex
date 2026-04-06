// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matcher for tensor-like extension types.

use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::fixed_shape::FixedShapeTensor;
use crate::fixed_shape::FixedShapeTensorMetadata;
use crate::vector::AnyVector;
use crate::vector::VectorMatcherMetadata;

/// Matcher for any tensor-like extension type.
///
/// Currently the different kinds of tensors that are available are:
///
/// - `FixedShapeTensor`
/// - `Vector`
pub struct AnyTensor;

/// The matched variant of a tensor-like extension type.
#[derive(Debug, PartialEq, Eq)]
pub enum TensorMatch<'a> {
    /// A [`FixedShapeTensor`] extension type.
    FixedShapeTensor(&'a FixedShapeTensorMetadata),

    /// A [`Vector`] extension type.
    ///
    /// Note that we store an owned type here wrapping (copyable) data from the dtype.
    Vector(VectorMatcherMetadata),
}

impl Matcher for AnyTensor {
    type Match<'a> = TensorMatch<'a>;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(metadata) = ext_dtype.metadata_opt::<FixedShapeTensor>() {
            return Some(TensorMatch::FixedShapeTensor(metadata));
        }

        // Special logic for vectors to get convenience metadata (instead of `EmptyMetadata`).
        if let Some(metadata) = ext_dtype.metadata_opt::<AnyVector>() {
            return Some(TensorMatch::Vector(metadata));
        }

        None
    }
}
