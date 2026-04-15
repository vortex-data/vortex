// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types and functionality for working with tensors, vectors, and related mathematical constructs
//! including unit vectors, spherical coordinates, and similarity measures such as cosine
//! similarity.

use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

use crate::fixed_shape::FixedShapeTensor;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::scalar_fns::sorf_transform::SorfTransform;
use crate::vector::Vector;

pub mod matcher;
pub mod scalar_fns;

pub mod fixed_shape;
pub mod vector;

pub mod encodings;

pub mod vector_search;

mod utils;

/// Initialize the Vortex tensor library with a Vortex session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Vector);
    session.dtypes().register(FixedShapeTensor);

    let session_fns = session.scalar_fns();
    let session_arrays = session.arrays();

    session_fns.register(CosineSimilarity);
    session_fns.register(InnerProduct);
    session_fns.register(L2Denorm);
    session_fns.register(L2Norm);
    session_fns.register(SorfTransform);

    session_arrays.register(ScalarFnArrayPlugin::new(CosineSimilarity));
    session_arrays.register(ScalarFnArrayPlugin::new(InnerProduct));
    session_arrays.register(ScalarFnArrayPlugin::new(L2Denorm));
    session_arrays.register(ScalarFnArrayPlugin::new(L2Norm));
    session_arrays.register(ScalarFnArrayPlugin::new(SorfTransform));
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
