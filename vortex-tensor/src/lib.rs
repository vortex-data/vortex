// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types and functionality for working with tensors, vectors, and related mathematical constructs
//! including unit vectors, spherical coordinates, and similarity measures such as cosine
//! similarity.

use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

use crate::fixed_shape::FixedShapeTensor;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::vector::Vector;

pub mod matcher;
pub mod scalar_fns;

pub mod fixed_shape;
pub mod vector;

pub mod encodings;

mod utils;

/// Registers the tensor extension dtypes and scalar functions with the given session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Vector);
    session.dtypes().register(FixedShapeTensor);
    session.scalar_fns().register(CosineSimilarity);
    session.scalar_fns().register(InnerProduct);
    session.scalar_fns().register(L2Norm);
}
