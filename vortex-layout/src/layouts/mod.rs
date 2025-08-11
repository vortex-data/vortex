// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of built-in layouts for Vortex

use crate::LazyWithSegments;
use vortex_array::ArrayRef;

pub mod buffered;
pub mod chunked;
#[cfg(feature = "zstd")]
pub mod compact;
pub mod compressed;
pub mod dict;
pub mod file_stats;
pub mod flat;
mod partitioned;
pub mod repartition;
pub mod row_idx;
pub mod struct_;
pub mod zoned;

type SharedArray = LazyWithSegments<ArrayRef>;
