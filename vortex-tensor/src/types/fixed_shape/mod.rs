// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

/// Vortex extension id for [`FixedShapeTensor`].
pub(crate) const ID: &str = "vortex.tensor.fixed_shape_tensor";

/// Arrow canonical extension name [`ID`] aliases to.
pub(crate) const ARROW_EXT_NAME: &str = "arrow.fixed_shape_tensor";

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

mod matcher;
pub use matcher::AnyFixedShapeTensor;
pub use matcher::FixedShapeTensorMatcherMetadata;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod canonical;
mod vtable;
