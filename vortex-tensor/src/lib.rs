// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types and functionality for working with tensors, vectors, and related mathematical constructs
//! including unit vectors, spherical coordinates, and similarity measures such as cosine
//! similarity.

#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::unwrap_in_result)
)]

use std::sync::Arc;

use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::arrow::ArrowSession;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_session::VortexSessionBuilder;

use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::vector::Vector;

pub mod matcher;
pub mod scalar_fns;

mod types;

pub use types::fixed_shape_tensor;
pub use types::vector;

pub mod encodings;

pub mod vector_search;

mod utils;

/// Environment variable that gates registration of the tensor scalar-fn array plugins (the array
/// encodings that let [`CosineSimilarity`], [`InnerProduct`], [`L2Denorm`], and [`L2Norm`]
/// persist in a Vortex file). When unset, only the scalar functions themselves
/// are registered; readers of files containing serialized tensor scalar-fn arrays will fail to
/// deserialize. Opt-in by setting the variable to any non-empty value.
pub const SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV: &str = "VX_SCALAR_FN_ARRAY_TENSOR_PLUGIN";

/// Initialize the Vortex tensor library with a Vortex session builder.
pub fn initialize(session: &mut VortexSessionBuilder) {
    {
        let dtypes = session.get_mut::<DTypeSession>();
        dtypes.register(Vector);
        dtypes.register(FixedShapeTensor);
    }

    {
        let arrow_session = session.get_mut::<ArrowSession>();
        arrow_session.register_exporter(Arc::new(Vector));
        arrow_session.register_importer(Arc::new(Vector));
    }

    {
        let scalar_fns = session.get_mut::<ScalarFnSession>();
        scalar_fns.register(CosineSimilarity);
        scalar_fns.register(InnerProduct);
        scalar_fns.register(L2Denorm);
        scalar_fns.register(L2Norm);
    }

    // Registering the scalar-fn array plugins lets the tensor scalar fns be serialized as array
    // encodings inside Vortex files. Gate this on an env var so applications that do not intend
    // to persist these encodings do not pay the registry cost or widen their stable-encoding
    // surface unintentionally.
    if std::env::var_os(SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV).is_some_and(|v| !v.is_empty()) {
        let arrays = session.get_mut::<ArraySession>();

        arrays.register(ScalarFnArrayPlugin::new(CosineSimilarity));
        arrays.register(ScalarFnArrayPlugin::new(InnerProduct));
        arrays.register(ScalarFnArrayPlugin::new(L2Denorm));
        arrays.register(ScalarFnArrayPlugin::new(L2Norm));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let mut builder = vortex_array::default_session_builder();
        crate::initialize(&mut builder);
        builder.build()
    });
}
