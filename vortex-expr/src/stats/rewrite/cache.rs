// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A session-scoped cache of stats rewrites.

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactKey;
use vortex_array::expr::WeakBoundExprKey;
use vortex_array::scalar_fn::fns::dynamic_expression::DynamicExpression;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::rewrite::dynamic_expression::StatsProof;

/// The maximum number of rewrites that the cache keeps for each proof kind. When the live entries
/// fill the cache, it removes all entries of that kind before it adds a new entry.
const MAX_ENTRIES: usize = 4096;

type Rewrites = HashMap<WeakBoundExprKey, Option<BoundExpression>>;

/// A cache of stats rewrites, keyed by the identity of the predicate tree.
///
/// A scan binds its filter once against the table schema, and gives the same tree to the reader
/// of each file. Thus an identity key gets a hit for each file after the first, and the key
/// comparison does not read the schema.
///
/// The keys hold their predicates weakly. When a scan drops its filter, no lookup can hit its
/// entries again, and the next insert removes them.
#[derive(Default)]
pub(crate) struct RewriteCache {
    /// Increments each time that a rule is registered. A rewrite that started before can use old
    /// rules, so the cache does not keep it.
    revision: u64,
    falsify: Rewrites,
    satisfy: Rewrites,
}

impl Debug for RewriteCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewriteCache")
            .field("revision", &self.revision)
            .field("falsify", &self.falsify.len())
            .field("satisfy", &self.satisfy.len())
            .finish()
    }
}

impl RewriteCache {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the cached rewrite of `expr`, or `None` if the cache does not hold it.
    pub(crate) fn get(
        &self,
        proof: StatsProof,
        expr: &BoundExpression,
    ) -> Option<Option<BoundExpression>> {
        self.rewrites(proof).get(expr as &dyn ExactKey).cloned()
    }

    /// Adds the rewrite of `expr`, if no rule was registered after `revision` was read. Removes
    /// the entries of predicates that are no longer alive.
    pub(crate) fn insert(
        &mut self,
        revision: u64,
        proof: StatsProof,
        expr: &BoundExpression,
        rewrite: Option<BoundExpression>,
    ) {
        if self.revision != revision {
            return;
        }

        let rewrites = self.rewrites_mut(proof);
        rewrites.retain(|key, _| key.is_live());
        if rewrites.len() >= MAX_ENTRIES {
            rewrites.clear();
        }

        rewrites.insert(WeakBoundExprKey::new(expr), rewrite);
    }

    /// Removes all entries, and stops the rewrites that are in progress from being added.
    pub(crate) fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.falsify.clear();
        self.satisfy.clear();
    }

    fn rewrites(&self, proof: StatsProof) -> &Rewrites {
        match proof {
            StatsProof::Falsify => &self.falsify,
            StatsProof::Satisfy => &self.satisfy,
        }
    }

    fn rewrites_mut(&mut self, proof: StatsProof) -> &mut Rewrites {
        match proof {
            StatsProof::Falsify => &mut self.falsify,
            StatsProof::Satisfy => &mut self.satisfy,
        }
    }
}

/// Returns `true` if the cache can keep the rewrite of `expr`.
///
/// The proof of a dynamic expression holds the session. If the session cache kept the proof, the
/// session would not be released. The static subexpressions of such a predicate are still
/// cached, because their rewrites are recursive calls.
pub(crate) fn is_cacheable(expr: &BoundExpression) -> bool {
    !expr.is::<DynamicExpression>() && expr.children().iter().all(is_cacheable)
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn insert_drops_rewrites_of_dropped_predicates() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut cache = RewriteCache::default();

        let finished = not(root()).bind(&dtype)?;
        let proof = Some(lit(true).bind(&dtype)?);
        cache.insert(0, StatsProof::Falsify, &finished, proof.clone());
        assert_eq!(cache.get(StatsProof::Falsify, &finished), Some(proof));
        drop(finished);

        let running = not(root()).bind(&dtype)?;
        cache.insert(0, StatsProof::Falsify, &running, None);
        assert_eq!(cache.falsify.len(), 1);
        assert_eq!(cache.get(StatsProof::Falsify, &running), Some(None));
        Ok(())
    }

    #[test]
    fn insert_skips_rewrites_of_an_old_revision() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut cache = RewriteCache::default();
        let revision = cache.revision();
        cache.invalidate();

        let expr = not(root()).bind(&dtype)?;
        cache.insert(revision, StatsProof::Satisfy, &expr, None);
        assert_eq!(cache.get(StatsProof::Satisfy, &expr), None);
        Ok(())
    }
}
