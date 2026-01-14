// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of built-in layouts for Vortex

use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::ArrayRef;
use vortex_error::SharedVortexResult;

pub mod buffered;
pub mod chunked;
pub mod collect;
#[cfg(feature = "zstd")]
pub mod compact;
pub mod compressed;
pub mod dict;
pub mod file_stats;
pub mod flat;
pub(crate) mod partitioned;
pub mod repartition;
pub mod row_idx;
pub mod struct_;
pub mod table;
pub mod zoned;

pub type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;
