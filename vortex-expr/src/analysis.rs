// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::stats::Stat;
use vortex_dtype::FieldPath;

use crate::ExprRef;

/// A catalog of available stats that are associated with field paths.
pub trait StatsCatalog {
    /// Given a field path and statistic, return an expression that when evaluated over the catalog
    /// will return that stat for the referenced field.
    ///
    /// This is likely to be a column expression, or a literal.
    ///
    /// Returns `None` if the stat is not available for the field path.
    fn stats_ref(&mut self, _field_path: &FieldPath, _stat: Stat) -> Option<ExprRef> {
        None
    }
}

/// This can be used by expression to plug into vortex expression analysis, such as
/// pruning or expression simplification
pub trait AnalysisExpr {
    /// An expression over zone-statistics which implies all records in the zone evaluate to false.
    ///
    /// Given an expression, `e`, if `e.stat_falsification(..)` evaluates to true, it is guaranteed
    /// that `e` evaluates to false on all records in the zone. However, the inverse is not
    /// necessarily true: even if the falsification evaluates to false, `e` need not evaluate to
    /// true on all records.
    ///
    /// The [`StatsCatalog`] can be used to constrain or rename stats used in the final expr.
    ///
    /// # Examples
    ///
    /// - An expression over one variable: `x > 0` is false for all records in a zone if the maximum
    ///   value of the column `x` in that zone is less than or equal to zero: `max(x) <= 0`.
    /// - An expression over two variables: `x > y` becomes `max(x) <= min(y)`.
    /// - A conjunctive expression: `x > y AND z < x` becomes `max(x) <= min(y) OR min(z) >= max(x).
    ///
    /// Some expressions, in theory, have falsifications but this function does not support them
    /// such as `x < (y < z)` or `x LIKE "needle%"`.
    fn stat_falsification(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    /// An expression for the upper non-null bound of this expression, if available.
    ///
    /// This function returns None if there is no upper bound or it is difficult to compute.
    ///
    /// The returned expression evaluates to null if the maximum value is unknown. In that case, you
    /// _must not_ assume the array is empty _nor_ may you assume the array only contains non-null
    /// values.
    fn max(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    /// An expression for the lower non-null bound of this expression, if available.
    ///
    /// See [AnalysisExpr::max] for important details.
    fn min(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    /// An expression for the NaN count for a column, if available.
    ///
    /// This method returns `None` if the NaNCount stat is unknown.
    fn nan_count(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    fn field_path(&self) -> Option<FieldPath> {
        None
    }

    // TODO: add containment
}
