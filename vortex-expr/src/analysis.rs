use vortex_array::stats::Stat;
use vortex_dtype::FieldPath;

use crate::{ExprRef, Identifier};

pub trait StatsCatalog {
    /// Given an id, field and stat return an expression that when evaluated will return that stat
    /// this would be a column reference or a literal value, if the value is known at planning time.
    fn stats_ref(&mut self, _id: &Identifier, _field: &FieldPath, _stat: Stat) -> Option<ExprRef> {
        None
    }
}

/// This can be used by expression to plug into vortex expression analysis, such as
/// pruning or expression simplification
pub trait AnalysisExpr {
    /// Tries to find an expression over zone-statistics which is true if-and-only-if `expr` is
    /// false for all records in the zone.
    ///
    /// The `StatsCatalog` can be used to constrain or rename stats used in the final expr.
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

    /// If an expression is returned, its value is an upper bound on the value of `expr`.
    ///
    /// We may return `None` for values which have no upper bound or values for which knowing the
    /// upper bound is difficult.
    fn max(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }
    /// If an expression is returned, its value is an upper bound on the value of `expr`.
    /// see `AnalysisExpr::max`
    fn min(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    fn field_path(&self) -> Option<(Identifier, FieldPath)> {
        None
    }

    // TODO: add containment
}
