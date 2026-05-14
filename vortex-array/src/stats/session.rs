// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for stats rewrite rules.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scalar_fn::ScalarFnId;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::StatsRewriteRuleRef;

type StatsRewriteRuleSet = Arc<[StatsRewriteRuleRef]>;

/// Session state for stats rewrite rules.
#[derive(Debug, Default)]
pub struct StatsRewriteSession {
    rules: RwLock<HashMap<ScalarFnId, StatsRewriteRuleSet>>,
}

impl StatsRewriteSession {
    /// Register a stats rewrite rule.
    #[allow(dead_code)]
    pub(crate) fn register<R: StatsRewriteRule>(&self, rule: R) {
        self.register_ref(Arc::new(rule));
    }

    /// Register a shared stats rewrite rule.
    #[allow(dead_code)]
    pub(crate) fn register_ref(&self, rule: StatsRewriteRuleRef) {
        let mut rules = self.rules.write();
        let rule_id = rule.scalar_fn_id();
        let mut updated_rules = rules
            .get(&rule_id)
            .map(|rules| rules.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        updated_rules.push(rule);
        rules.insert(rule_id, updated_rules.into());
    }

    /// Return the rewrite rules registered for `scalar_fn_id`.
    pub(crate) fn rules_for(&self, scalar_fn_id: ScalarFnId) -> Option<StatsRewriteRuleSet> {
        self.rules.read().get(&scalar_fn_id).cloned()
    }
}

impl SessionVar for StatsRewriteSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing stats rewrite session data.
pub(crate) trait StatsRewriteSessionExt: SessionExt {
    /// Returns the stats rewrite rule registry.
    fn stats_rewrites(&self) -> Ref<'_, StatsRewriteSession> {
        self.get::<StatsRewriteSession>()
    }
}
impl<S: SessionExt> StatsRewriteSessionExt for S {}
