// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod pruning_expr;
mod relation;

pub use pruning_expr::RequiredStats;
pub use pruning_expr::checked_pruning_expr;
pub use pruning_expr::field_path_stat_field_name;
pub use relation::Relation;

use crate::dtype::FieldPath;
use crate::expr::Expression;
use crate::expr::stats::Stat;

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
