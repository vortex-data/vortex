// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types and functionality for working with tensors, vectors, and related mathematical constructs
//! including unit vectors, spherical coordinates, and similarity measures such as cosine
//! similarity.

#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::unwrap_in_result)
)]

use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::scalar_fns::sorf_transform::SorfTransform;
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
/// encodings that let [`CosineSimilarity`], [`InnerProduct`], [`L2Denorm`], [`L2Norm`], and
/// [`SorfTransform`] persist in a Vortex file). When unset, only the scalar functions themselves
/// are registered; readers of files containing serialized tensor scalar-fn arrays will fail to
/// deserialize. Opt-in by setting the variable to any non-empty value.
pub const SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV: &str = "VX_SCALAR_FN_ARRAY_TENSOR_PLUGIN";

/// Initialize the Vortex tensor library with a Vortex session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Vector);
    session.dtypes().register(FixedShapeTensor);

    let session_fns = session.scalar_fns();

    session_fns.register(CosineSimilarity);
    session_fns.register(InnerProduct);
    session_fns.register(L2Denorm);
    session_fns.register(L2Norm);
    session_fns.register(SorfTransform);

    // Registering the scalar-fn array plugins lets the tensor scalar fns be serialized as array
    // encodings inside Vortex files. Gate this on an env var so applications that do not intend
    // to persist these encodings do not pay the registry cost or widen their stable-encoding
    // surface unintentionally.
    if std::env::var_os(SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV).is_some_and(|v| !v.is_empty()) {
        let session_arrays = session.arrays();

        session_arrays.register(ScalarFnArrayPlugin::new(CosineSimilarity));
        session_arrays.register(ScalarFnArrayPlugin::new(InnerProduct));
        session_arrays.register(ScalarFnArrayPlugin::new(L2Denorm));
        session_arrays.register(ScalarFnArrayPlugin::new(L2Norm));
        session_arrays.register(ScalarFnArrayPlugin::new(SorfTransform));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::session::ArraySession;
    use vortex_session::VortexSession;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    });
}
