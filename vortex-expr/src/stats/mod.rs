// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics-based pruning: stat expressions, binding them to available statistics, and the
//! falsify/satisfy rewrite rules that turn predicates into pruning predicates.

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
pub use session::*;

pub mod bind;
pub mod expr;
pub mod rewrite;
pub mod session;
pub mod stat_fn;
