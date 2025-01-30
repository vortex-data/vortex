use std::fmt::Debug;

use datafusion_common::stats::Precision;
use vortex_array::stats;
use vortex_array::stats::DirectionalBound;

pub fn directional_bound_to_df_precision<T: Debug + Clone + Eq + PartialOrd>(
    bound: Option<DirectionalBound<T>>,
) -> Precision<T> {
    match bound.map(|bound| bound.value()) {
        Some(stats::Precision::Exact(val)) => Precision::Exact(val),
        Some(stats::Precision::Bound(val)) => Precision::Inexact(val),
        None => Precision::Absent,
    }
}
