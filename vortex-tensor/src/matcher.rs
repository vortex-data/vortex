// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matcher for tensor-like extension types.

use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::types::fixed_shape::AnyFixedShapeTensor;
use crate::types::fixed_shape::FixedShapeTensorMatcherMetadata;
use crate::types::normalized_vector::AnyNormalizedVector;
use crate::types::vector::AnyVector;
use crate::types::vector::VectorMatcherMetadata;

/// Matcher for any tensor-like extension type.
///
/// Currently the different kinds of tensors that are available are:
///
/// - `FixedShapeTensor`
/// - `Vector`
/// - `NormalizedVector`
pub struct AnyTensor;

/// The matched variant of a tensor-like extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMatch<'a> {
    /// A [`FixedShapeTensor`](crate::fixed_shape::FixedShapeTensor) extension type.
    FixedShapeTensor(FixedShapeTensorMatcherMetadata<'a>),

    /// A [`Vector`](crate::vector::Vector) extension type.
    ///
    /// Note that we store an owned type here wrapping (copyable) data from the dtype.
    Vector(VectorMatcherMetadata),

    /// A [`NormalizedVector`](crate::normalized_vector::NormalizedVector) extension over
    /// [`Vector`](crate::vector::Vector) storage.
    NormalizedVector(VectorMatcherMetadata),
}

impl TensorMatch<'_> {
    /// Returns the tensor element type for this tensor-like dtype.
    pub fn element_ptype(self) -> PType {
        match self {
            Self::FixedShapeTensor(metadata) => metadata.element_ptype(),
            Self::Vector(metadata) | Self::NormalizedVector(metadata) => metadata.element_ptype(),
        }
    }

    /// Returns the flattened element count for each logical tensor row.
    pub fn list_size(self) -> u32 {
        match self {
            Self::FixedShapeTensor(metadata) => metadata.flat_list_size(),
            Self::Vector(metadata) | Self::NormalizedVector(metadata) => metadata.dimensions(),
        }
    }

    /// Returns `true` when the dtype is a
    /// [`NormalizedVector`](crate::normalized_vector::NormalizedVector).
    pub fn is_normalized(self) -> bool {
        matches!(self, Self::NormalizedVector(_))
    }
}

impl Matcher for AnyTensor {
    type Match<'a> = TensorMatch<'a>;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(metadata) = ext_dtype.metadata_opt::<AnyFixedShapeTensor>() {
            return Some(TensorMatch::FixedShapeTensor(metadata));
        }

        // Check `AnyNormalizedVector` first because `AnyVector` is inclusive: it would otherwise
        // match `NormalizedVector` and we'd lose the normalized variant in the returned
        // `TensorMatch`.
        if let Some(metadata) = ext_dtype.metadata_opt::<AnyNormalizedVector>() {
            return Some(TensorMatch::NormalizedVector(metadata));
        }

        if let Some(metadata) = ext_dtype.metadata_opt::<AnyVector>() {
            return Some(TensorMatch::Vector(metadata));
        }

        None
    }
}
