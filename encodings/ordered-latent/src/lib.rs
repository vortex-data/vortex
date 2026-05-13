// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's order-preserving recast from primitive
//! `T` to an unsigned latent `L`.
//!
//! See [`OrderedLatentArray`] for the public type.

pub use array::*;

mod array;
mod transforms;
