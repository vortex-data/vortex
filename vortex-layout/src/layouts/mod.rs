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
pub mod struct_;
pub mod zoned;

type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;
