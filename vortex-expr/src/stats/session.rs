// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for stats APIs.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::derive::StatDerivation;
use crate::stats::derive::StatDerivationRef;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::StatsRewriteRuleRef;
use crate::stats::rewrite::register_builtins;

type StatsRewriteRuleSet = Arc<[StatsRewriteRuleRef]>;

/// Session state for stats APIs.
#[derive(Clone, Debug)]
pub struct StatsSession {
    rewrite_rules: Arc<RwLock<HashMap<ScalarFnId, StatsRewriteRuleSet>>>,
    stat_derivations: Arc<RwLock<HashMap<ScalarFnId, StatDerivationRef>>>,
}

impl Default for StatsSession {
    fn default() -> Self {
        let this = Self {
            rewrite_rules: Arc::new(RwLock::new(HashMap::default())),
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
        let mut rules = self.rewrite_rules.write();
        let rule_id = rule.scalar_fn_id();
        let mut updated_rules = rules
            .get(&rule_id)
            .map(|rules| rules.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        updated_rules.push(rule);
        rules.insert(rule_id, updated_rules.into());
    }

    /// Return the rewrite rules registered for `scalar_fn_id`.
    pub(crate) fn rewrite_rules_for(
        &self,
        scalar_fn_id: ScalarFnId,
    ) -> Option<StatsRewriteRuleSet> {
        self.rewrite_rules.read().get(&scalar_fn_id).cloned()
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
        self.stat_derivations
            .write()
            .insert(derivation.scalar_fn_id(), derivation);
    }

    /// Return the statistic derivation registered for `scalar_fn_id`.
    pub(crate) fn stat_derivation_for(
        &self,
        scalar_fn_id: ScalarFnId,
    ) -> Option<StatDerivationRef> {
        self.stat_derivations.read().get(&scalar_fn_id).cloned()
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
