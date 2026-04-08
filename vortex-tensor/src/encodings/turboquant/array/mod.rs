// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms, centroids (codebook),
//! and rotation signs.

pub(crate) mod data;
pub(crate) mod slots;

pub(crate) mod centroids;
pub(crate) mod rotation;
