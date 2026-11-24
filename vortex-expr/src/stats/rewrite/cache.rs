// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A session-scoped cache of stats rewrites.

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactKey;
use vortex_array::expr::WeakBoundExprKey;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::rewrite::StatsProof;

/// Number of inserts between dead-key eviction sweeps.
const EVICT_INTERVAL: usize = 64;

/// The maximum number of rewrites that the cache keeps for each proof kind. When the live entries
/// fill the cache, it removes all entries of that kind before it adds a new entry.
const MAX_ENTRIES: usize = 4096;

#[derive(Default)]
struct ProofCache {
    entries: HashMap<WeakBoundExprKey, Option<BoundExpression>>,
    inserts: usize,
}

impl ProofCache {
    fn contains(&self, expr: &BoundExpression) -> bool {
        self.entries.contains_key(expr as &dyn ExactKey)
    }

    fn get(&self, expr: &BoundExpression) -> Option<Option<BoundExpression>> {
        self.entries.get(expr as &dyn ExactKey).cloned()
    }

    fn insert(&mut self, expr: &BoundExpression, rewrite: Option<BoundExpression>) {
        self.inserts += 1;
        if self.inserts >= EVICT_INTERVAL || self.entries.len() >= MAX_ENTRIES {
            self.inserts = 0;
            self.entries.retain(|key, _| key.is_live());
            if self.entries.len() >= MAX_ENTRIES {
                self.entries.clear();
            }
        }
        self.entries.insert(WeakBoundExprKey::new(expr), rewrite);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.inserts = 0;
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// A cache of stats rewrites, keyed by the identity of the predicate tree.
///
/// A scan binds its filter once against the table schema, and gives the same tree to the reader
/// of each file. Thus an identity key gets a hit for each file after the first, and the key
/// comparison does not read the schema.
///
/// The keys hold their predicates weakly. When a scan drops its filter, no lookup can hit its
/// entries again, and periodic eviction sweeps drop them.
#[derive(Default)]
pub(crate) struct RewriteCache {
    /// Increments each time that a rule is registered. A rewrite that started before can use old
    /// rules, so the cache does not keep it.
    revision: u64,
    falsify: ProofCache,
    satisfy: ProofCache,
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
        self.rewrites(proof).get(expr)
    }

    /// Returns `true` if the cache holds a rewrite for `expr`.
    pub(crate) fn contains(&self, proof: StatsProof, expr: &BoundExpression) -> bool {
        self.rewrites(proof).contains(expr)
    }

    /// Adds the rewrite of `expr`, if no rule was registered after `revision` was read.
    /// Periodically removes the entries of predicates that are no longer alive.
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

        self.rewrites_mut(proof).insert(expr, rewrite);
    }

    /// Removes all entries, and stops the rewrites that are in progress from being added.
    pub(crate) fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.falsify.clear();
        self.satisfy.clear();
    }

    fn rewrites(&self, proof: StatsProof) -> &ProofCache {
        match proof {
            StatsProof::Falsify => &self.falsify,
            StatsProof::Satisfy => &self.satisfy,
        }
    }

    fn rewrites_mut(&mut self, proof: StatsProof) -> &mut ProofCache {
        match proof {
            StatsProof::Falsify => &mut self.falsify,
            StatsProof::Satisfy => &mut self.satisfy,
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::*;
    use crate::lit;
    use crate::not;
    use crate::root;

    #[test]
    fn insert_drops_rewrites_of_dropped_predicates() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut cache = RewriteCache::default();

        let finished = not(root()).bind(&dtype)?;
        let proof = Some(lit(true).bind(&dtype)?);
        cache.insert(0, StatsProof::Falsify, &finished, proof.clone());
        assert_eq!(cache.get(StatsProof::Falsify, &finished), Some(proof));
        drop(finished);

        // Inserts before the eviction interval do not sweep dead keys.
        let mut live = Vec::with_capacity(EVICT_INTERVAL - 2);
        for _ in 0..EVICT_INTERVAL - 2 {
            let expr = not(root()).bind(&dtype)?;
            cache.insert(0, StatsProof::Falsify, &expr, None);
            live.push(expr);
        }
        assert_eq!(cache.falsify.len(), EVICT_INTERVAL - 1);

        // The EVICT_INTERVAL-th insert sweeps the dropped key while keeping live ones.
        let running = not(root()).bind(&dtype)?;
        cache.insert(0, StatsProof::Falsify, &running, None);
        assert_eq!(cache.falsify.len(), EVICT_INTERVAL - 1);
        assert_eq!(cache.get(StatsProof::Falsify, &running), Some(None));
        drop(live);
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
