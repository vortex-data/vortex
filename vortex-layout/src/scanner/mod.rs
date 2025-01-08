mod scan;

use std::fmt::Debug;

pub use scan::*;
use vortex_array::stats::StatsSet;
use vortex_array::ArrayData;

use crate::operations::Operation;
use crate::segments::SegmentReader;
pub type EvalOp = Box<dyn Operation<Output = ArrayData>>;
pub type StatsOp = Box<dyn Operation<Output = Vec<StatsSet>>>;
