// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod annotations;
pub mod immediate_access;
mod null_sensitive;

pub use annotations::*;
pub use immediate_access::*;
pub use null_sensitive::*;
use vortex_dtype::FieldPath;

use crate::expr::Expression;
use crate::stats::Stat;

/// A catalog of available stats that are associated with field paths.
pub trait StatsCatalog {
    /// Given a field path and statistic, return an expression that when evaluated over the catalog
    /// will return that stat for the referenced field.
    ///
    /// This is likely to be a column expression, or a literal.
    ///
    /// Returns `None` if the stat is not available for the field path.
    fn stats_ref(&self, _field_path: &FieldPath, _stat: Stat) -> Option<Expression> {
        None
    }
}
