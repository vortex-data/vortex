// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matcher for tensor-like extension types.

use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::types::fixed_shape_tensor::AnyFixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMatcherMetadata;
use crate::types::vector::AnyVector;
use crate::types::vector::VectorMatcherMetadata;

/// Matcher for any tensor-like extension type.
///
/// Currently the different kinds of tensors that are available are:
///
/// - `FixedShapeTensor`
/// - `Vector`
pub struct AnyTensor;

/// The matched variant of a tensor-like extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMatch<'a> {
    /// A [`FixedShapeTensor`](crate::fixed_shape_tensor::FixedShapeTensor) extension type.
    FixedShapeTensor(FixedShapeTensorMatcherMetadata<'a>),

    /// A [`Vector`](crate::vector::Vector) extension type.
    ///
    /// Note that we store an owned type here wrapping (copyable) data from the dtype.
    Vector(VectorMatcherMetadata),
}

impl TensorMatch<'_> {
    /// Returns the tensor element type for this tensor-like dtype.
    pub fn element_ptype(self) -> PType {
        match self {
            Self::FixedShapeTensor(metadata) => metadata.element_ptype(),
            Self::Vector(metadata) => metadata.element_ptype(),
        }
    }

    /// Returns the flattened element count for each logical tensor row.
    pub fn list_size(self) -> u32 {
        match self {
            Self::FixedShapeTensor(metadata) => metadata.flat_list_size(),
            Self::Vector(metadata) => metadata.dimensions(),
        }
    }
}

impl Matcher for AnyTensor {
    type Match<'a> = TensorMatch<'a>;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(metadata) = ext_dtype.metadata_opt::<AnyFixedShapeTensor>() {
            return Some(TensorMatch::FixedShapeTensor(metadata));
        }

        // Special logic for vectors to get convenience metadata (instead of `EmptyMetadata`).
        if let Some(metadata) = ext_dtype.metadata_opt::<AnyVector>() {
            return Some(TensorMatch::Vector(metadata));
        }

        None
    }
}
