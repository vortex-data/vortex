// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

use vortex_array::dtype::extension::ExtId;
use vortex_session::registry::CachedId;

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

impl FixedShapeTensor {
    pub(crate) fn arrow_ext_id() -> ExtId {
        static ID: CachedId = CachedId::new("arrow.fixed_shape_tensor");
        *ID
    }
}

mod matcher;
pub use matcher::AnyFixedShapeTensor;
pub use matcher::FixedShapeTensorMatcherMetadata;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod canonical;
mod proto;
mod vtable;
pub(crate) use canonical::json_to_proto;
pub(crate) use canonical::proto_to_json;
