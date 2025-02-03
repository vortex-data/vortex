use std::fmt::Debug;

use datafusion_common::stats::Precision;
use vortex_array::stats;

// TODO(joe + gatesn): Move datafusion convert logic here.

pub fn directional_bound_to_df_precision<T: Debug + Clone + Eq + PartialOrd>(
    bound: Option<stats::Precision<T>>,
) -> Precision<T> {
    match bound {
        Some(stats::Precision::Exact(val)) => Precision::Exact(val),
        Some(stats::Precision::Inexact(val)) => Precision::Inexact(val),
        None => Precision::Absent,
    }
}
