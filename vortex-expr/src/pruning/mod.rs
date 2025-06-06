mod field_or_identity;
mod pruning_predicate;
mod relation;

pub use field_or_identity::{FieldOrIdentity, stat_field_name};
pub use pruning_predicate::PruningPredicate;
use vortex_array::stats::Stat;
use vortex_dtype::FieldPath;

use crate::{ExprRef, Identifier};

pub trait StatsCatalog {
    // Given an id, field and stat return an expression that when evaluated will return that stat
    // this would be a column reference or a literal value, if the value is known at planning time.
    // TODO(joe): replace field with a field path, once implemented.
    fn stats_ref(&mut self, _id: &Identifier, _field: &FieldPath, _stat: Stat) -> Option<ExprRef> {
        None
    }
}

pub trait AnalysisExpr {
    fn prune_expr(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    fn max(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    fn min(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        None
    }

    fn field_path(&self) -> Option<(Identifier, FieldPath)> {
        None
    }

    // TODO: add containment
}
