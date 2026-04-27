// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

impl FixedShapeTensor {
    /// Arrow canonical extension name aliased to this type's [`ExtVTable::id`].
    ///
    /// [`ExtVTable::id`]: vortex_array::dtype::extension::ExtVTable::id
    pub(crate) const ARROW_EXT_NAME: &'static str = "arrow.fixed_shape_tensor";
}

mod matcher;
pub use matcher::AnyFixedShapeTensor;
pub use matcher::FixedShapeTensorMatcherMetadata;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod canonical;
mod vtable;
