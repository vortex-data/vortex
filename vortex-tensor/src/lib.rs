// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types and functionality for working with tensors, vectors, and related mathematical constructs
//! including unit vectors, spherical coordinates, and similarity measures such as cosine
//! similarity.

use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

use crate::encodings::turboquant::TurboQuant;
use crate::fixed_shape::FixedShapeTensor;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::vector::Vector;

pub mod matcher;
pub mod scalar_fns;

pub mod fixed_shape;
pub mod vector;

pub mod encodings;

mod utils;

/// Initialize the Vortex tensor library with a Vortex session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Vector);
    session.dtypes().register(FixedShapeTensor);

    session.arrays().register(TurboQuant);

    session.scalar_fns().register(CosineSimilarity);
    session.scalar_fns().register(InnerProduct);
    session.scalar_fns().register(L2Denorm);
    session.scalar_fns().register(L2Norm);
}
