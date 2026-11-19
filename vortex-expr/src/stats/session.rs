// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for stats APIs.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::derive::StatDerivation;
use crate::stats::derive::StatDerivationRef;
use crate::stats::rewrite::RewriteCache;
use crate::stats::rewrite::StatsProof;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::StatsRewriteRuleRef;
use crate::stats::rewrite::register_builtins;

pub(crate) type StatsRewriteRuleSet = Arc<[StatsRewriteRuleRef]>;

/// The result of a lookup in the session rewrite cache.
#[derive(Debug)]
pub(crate) enum RewriteLookup {
    /// The cached rewrite.
    Cached(Option<BoundExpression>),
    /// The cache does not hold the rewrite. Give `revision` to [`StatsSession::cache_rewrite`].
    Uncached {
        revision: u64,
        rules: Option<StatsRewriteRuleSet>,
    },
}

/// Session state for stats APIs.
#[derive(Clone, Debug)]
pub struct StatsSession {
    rewrite_rules: ArcSwapMap<ScalarFnId, StatsRewriteRuleSet>,
    rewrite_cache: Arc<RwLock<RewriteCache>>,
    stat_derivations: Arc<RwLock<HashMap<ScalarFnId, StatDerivationRef>>>,
}

impl Default for StatsSession {
    fn default() -> Self {
        let this = Self {
            rewrite_rules: ArcSwapMap::default(),
            rewrite_cache: Arc::new(RwLock::new(RewriteCache::default())),
            stat_derivations: Arc::new(RwLock::new(HashMap::default())),
        };
        register_builtins(&this);
        this
    }
}

impl StatsSession {
    /// Register a stats rewrite rule.
    pub fn register_rewrite<R: StatsRewriteRule>(&self, rule: R) {
        self.register_rewrite_ref(Arc::new(rule));
    }

    /// Register a shared stats rewrite rule.
    pub fn register_rewrite_ref(&self, rule: StatsRewriteRuleRef) {
        // Change the rules while the cache is locked, so that a lookup reads the revision and the
        // rules of the same registration.
        let mut cache = self.rewrite_cache.write();
        self.rewrite_rules.push(rule.scalar_fn_id(), rule);
        cache.invalidate();
    }

    /// Return the rewrite rules registered for `scalar_fn_id`.
    pub(crate) fn rewrite_rules_for(
        &self,
        scalar_fn_id: ScalarFnId,
    ) -> Option<StatsRewriteRuleSet> {
        self.rewrite_rules.get(&scalar_fn_id)
    }

    pub(crate) fn is_cacheable_expr(&self, expr: &BoundExpression) -> bool {
        let cacheable_node = expr.as_scalar().is_none_or(|scalar_fn| {
            self.rewrite_rules_for(scalar_fn.id())
                .is_none_or(|rules| rules.iter().all(|rule| rule.is_cacheable()))
        });
        cacheable_node
            && expr
                .children()
                .iter()
                .all(|child| self.is_cacheable_expr(child))
    }

    /// Returns the cached `proof` rewrite of `expr`, or the rules to make it and the cache
    /// revision of those rules.
    pub(crate) fn lookup_rewrite(
        &self,
        proof: StatsProof,
        expr: &BoundExpression,
        scalar_fn_id: ScalarFnId,
    ) -> RewriteLookup {
        let cache = self.rewrite_cache.read();
        if let Some(rewrite) = cache.get(proof, expr) {
            return RewriteLookup::Cached(rewrite);
        }

        RewriteLookup::Uncached {
            revision: cache.revision(),
            rules: self.rewrite_rules_for(scalar_fn_id),
        }
    }

    /// Adds the `proof` rewrite of `expr`, if no rule was registered after `revision`.
    pub(crate) fn cache_rewrite(
        &self,
        proof: StatsProof,
        revision: u64,
        expr: &BoundExpression,
        rewrite: Option<BoundExpression>,
    ) {
        self.rewrite_cache
            .write()
            .insert(revision, proof, expr, rewrite);
    }

    /// Register how the statistics of a scalar function's result derive from its inputs'.
    ///
    /// This replaces any derivation already registered for that function, including a builtin
    /// one, so a plugin can refine how a function's statistics are derived.
    pub fn register_stat_derivation<D: StatDerivation>(&self, derivation: D) {
        self.register_stat_derivation_ref(Arc::new(derivation));
    }

    /// Register a shared statistic derivation. See [`Self::register_stat_derivation`].
    pub fn register_stat_derivation_ref(&self, derivation: StatDerivationRef) {
        // Change the derivations while the cache is locked, so that a lookup reads the revision
        // and the derivations of the same registration.
        let mut cache = self.rewrite_cache.write();
        self.stat_derivations
            .write()
            .insert(derivation.scalar_fn_id(), derivation);
        cache.invalidate();
    }

    /// Return the statistic derivation registered for `scalar_fn_id`.
    pub(crate) fn stat_derivation_for(
        &self,
        scalar_fn_id: ScalarFnId,
    ) -> Option<StatDerivationRef> {
        self.stat_derivations.read().get(&scalar_fn_id).cloned()
    }

    /// Returns true if any rewrite rule is registered for `scalar_fn_id`.
    pub fn has_rewrite_for(&self, scalar_fn_id: ScalarFnId) -> bool {
        self.rewrite_rules.read().contains_key(&scalar_fn_id)
    }
}

impl SessionVar for StatsSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing stats session data.
pub trait StatsSessionExt: SessionExt {
    /// Returns the stats session state.
    fn stats(&self) -> SessionGuard<'_, StatsSession> {
        self.get::<StatsSession>()
    }
}
impl<S: SessionExt> StatsSessionExt for S {}
