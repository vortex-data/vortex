// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stored aggregate results and expression rewrites for pruning.
//!
//! File summaries expose finalized values through [`AggregateResults`]. Pruning expressions refer
//! to aggregate states, which file and zone binders resolve from compatible stored metadata.

mod results;
pub use results::AggregateResults;

pub mod bind;
pub mod compat;
pub mod expr;
pub use expr::all_nan;
pub use expr::all_non_nan;
pub use expr::all_non_null;
pub use expr::all_null;
pub use expr::bound;
pub use expr::min_max;
pub use expr::nan_count;
pub use expr::null_count;
pub use expr::stat;
pub use expr::sum;

pub mod rewrite;

pub mod session;
pub use session::*;

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;

/// Default file summaries. Numerical aggregates skip NaN values.
pub fn default_file_aggregates() -> Vec<AggregateFnRef> {
    vec![
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        Sum.bind(NumericalAggregateOpts::skip_nans()),
        NullCount.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
    ]
}
