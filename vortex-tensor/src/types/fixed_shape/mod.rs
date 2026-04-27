// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

impl FixedShapeTensor {
    pub(crate) const ARROW_EXT_NAME: &'static str = "arrow.fixed_shape_tensor";
}

mod matcher;
pub use matcher::AnyFixedShapeTensor;
pub use matcher::FixedShapeTensorMatcherMetadata;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod canonical;
mod proto;
mod vtable;
pub(crate) use canonical::{json_to_proto, proto_to_json};
