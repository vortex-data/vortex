// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vectors contain owned fixed-size canonical arrays of elements.
//!

// TODO(ngates): Currently, the data in a vector is Arc'd. We should consider whether we want the
//  performance hit for as_mut(), or whether we want zero-copy cloning. Not clear that we ever
//  need the clone behavior.

mod primitive;

use crate::pipeline::types::Canonical;
pub use primitive::*;
