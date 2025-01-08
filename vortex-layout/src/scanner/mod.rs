mod scan;

pub use scan::*;
use vortex_array::stats::StatsSet;
use vortex_array::ArrayData;

use crate::operations::Operation;

pub type EvalOp = Box<dyn Operation<Output = ArrayData>>;
pub type StatsOp = Box<dyn Operation<Output = Vec<StatsSet>>>;
