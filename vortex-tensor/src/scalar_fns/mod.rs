// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function expressions defined on tensor and tensor-like extension types.
//!
//! [`L2Norm`] returns row magnitudes, while [`L2Normalize`] returns normalized values together with
//! those magnitudes. [`InnerProduct`] and [`CosineSimilarity`] operate directly on decoded
//! coordinates.
//!
//! [`CosineSimilarity`]: cosine_similarity::CosineSimilarity
//! [`InnerProduct`]: inner_product::InnerProduct
//! [`L2Norm`]: l2_norm::L2Norm
//! [`L2Normalize`]: l2_normalize::L2Normalize

mod arithmetic;

pub mod cosine_similarity;
pub mod inner_product;
pub mod l2_norm;
pub mod l2_normalize;
