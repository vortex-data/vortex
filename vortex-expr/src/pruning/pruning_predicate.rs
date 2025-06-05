use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect as _, VortexResult};

use super::field_or_identity::FieldOrIdentity;
use super::pruning_predicate_builder::PruningPredicateBuilder;
use super::relation::Relation;
use crate::{ExprRef, Scope};

#[derive(Debug, Clone)]
pub struct PruningPredicate {
    pub(super) expr: ExprRef,
    pub(super) required_stats: Relation<FieldOrIdentity, Stat>,
}

impl PruningPredicate {
    pub fn try_new(original_expr: &ExprRef) -> Option<Self> {
        PruningPredicateBuilder::default().build(original_expr)
    }

    pub fn expr(&self) -> &ExprRef {
        &self.expr
    }

    pub fn required_stats(&self) -> &HashMap<FieldOrIdentity, HashSet<Stat>> {
        self.required_stats.map()
    }

    /// Evaluate this predicate against a per-chunk statistics table.
    ///
    /// Returns Ok(None) if any of the required statistics are not present in metadata.
    /// If it returns Ok(Some(array)), the array is a boolean array with the same length as the
    /// metadata, and a true value means the chunk _can_ be pruned.
    pub fn evaluate(&self, metadata: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        let known_stats = metadata
            .dtype()
            .as_struct()
            .vortex_expect("metadata must be struct array")
            .names()
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>();
        let required_stats = self
            .required_stats()
            .iter()
            .flat_map(|(key, value)| value.iter().map(|stat| key.stat_field_name_string(*stat)))
            .collect::<HashSet<_>>();
        let missing_stats = required_stats.difference(&known_stats).collect::<Vec<_>>();

        if !missing_stats.is_empty() {
            return Ok(None);
        }

        Ok(Some(self.expr.evaluate(&Scope::new(metadata.to_array()))?))
    }
}
