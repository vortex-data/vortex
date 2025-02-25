use std::fmt::Debug;

use datafusion_common::stats::Precision;
use vortex_array::stats;

// TODO(joe + gatesn): Move datafusion convert logic here.

pub fn directional_bound_to_df_precision<T: Debug + Clone + Eq + PartialOrd>(
    bound: Option<stats::Precision<T>>,
) -> Precision<T> {
    bound.map(bound_to_datafusion).unwrap_or_default()
}

pub fn bound_to_datafusion<T>(bound: stats::Precision<T>) -> Precision<T>
where
    T: Debug + Clone + Eq + PartialOrd,
{
    match bound {
        stats::Precision::Exact(val) => Precision::Exact(val),
        stats::Precision::Inexact(val) => Precision::Inexact(val),
    }
}
