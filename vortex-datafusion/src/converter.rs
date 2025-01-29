use std::fmt::Debug;

use datafusion_common::stats::Precision;
use vortex_array::stats;

pub fn precision_to_df_precision<T: Debug + Clone + Eq + PartialOrd>(
    precision: Option<stats::Precision<T>>,
) -> Precision<T> {
    match precision {
        Some(stats::Precision::Exact(val)) => Precision::Exact(val),
        Some(stats::Precision::Bound(val)) => Precision::Inexact(val),
        None => Precision::Absent,
    }
}
