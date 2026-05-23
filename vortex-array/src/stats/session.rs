// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for stats APIs.

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

/// Session state for stats APIs.
#[derive(Debug, Default)]
pub struct StatsSession {
    rewrite_rules: RwLock<HashMap<ScalarFnId, StatsRewriteRuleSet>>,
}

impl StatsSession {
    /// Register a stats rewrite rule.
    #[allow(dead_code)]
    pub(crate) fn register_rewrite<R: StatsRewriteRule>(&self, rule: R) {
        self.register_rewrite_ref(Arc::new(rule));
    }

    /// Register a shared stats rewrite rule.
    #[allow(dead_code)]
    pub(crate) fn register_rewrite_ref(&self, rule: StatsRewriteRuleRef) {
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
pub(crate) trait StatsSessionExt: SessionExt {
    /// Returns the stats session state.
    fn stats(&self) -> Ref<'_, StatsSession> {
        self.get::<StatsSession>()
    }
}
impl<S: SessionExt> StatsSessionExt for S {}
