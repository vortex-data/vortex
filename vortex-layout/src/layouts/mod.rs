//! A collection of built-in layouts for Vortex

use futures::future::{BoxFuture, Shared};
use vortex_array::ArrayRef;
use vortex_error::SharedVortexResult;

pub mod chunked;
pub mod dict;
pub mod file_stats;
pub mod filter;
pub mod flat;
pub mod repartition;
pub mod stats;
pub mod struct_;

type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;
