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
}

impl Default for StatsSession {
    fn default() -> Self {
        let this = Self {
            rewrite_rules: ArcSwapMap::default(),
            rewrite_cache: Arc::new(RwLock::new(RewriteCache::default())),
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
