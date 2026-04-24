// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

mod matcher;
pub use matcher::AnyFixedShapeTensor;
pub use matcher::FixedShapeTensorMatcherMetadata;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod proto;
mod vtable;
