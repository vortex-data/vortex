// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::extension::ExtDTypePluginRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::Timestamp;

/// Registry for extension dtypes.
pub type ExtDTypeRegistry = Registry<ExtDTypePluginRef>;

/// Bidirectional alias map between Vortex extension ids and Arrow canonical names.
type ArrowCanonicalMap = HashMap<ExtId, ExtId>;

/// Session for managing extension dtypes.
#[derive(Debug)]
pub struct DTypeSession {
    registry: ExtDTypeRegistry,
    arrow_canonical: ArcSwap<ArrowCanonicalMap>,
}

impl Default for DTypeSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
            arrow_canonical: ArcSwap::new(Arc::new(ArrowCanonicalMap::default())),
        };

        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        this
    }
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    pub fn register<V: ExtVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as ExtDTypePluginRef);
    }

    /// Return the registry of extension dtypes.
    pub fn registry(&self) -> &ExtDTypeRegistry {
        &self.registry
    }

    /// Alias an Arrow canonical extension name to a Vortex extension id. Aliased extensions
    /// emit the canonical name on `ARROW:extension:name` and serialize metadata as raw UTF-8
    /// instead of base64-wrapped bytes. Re-registering evicts the previous mapping.
    pub fn register_arrow_canonical(&self, vortex_id: ExtId, arrow_name: &'static str) {
        let arrow_id = ExtId::new(arrow_name);
        self.arrow_canonical.rcu(|prev| {
            let mut next = (**prev).clone();
            if let Some(stale) = next.insert(vortex_id, arrow_id) {
                next.remove(&stale);
            }
            if let Some(stale) = next.insert(arrow_id, vortex_id) {
                next.remove(&stale);
            }
            Arc::new(next)
        });
    }

    /// Returns the Arrow canonical extension name aliased to the given Vortex id, if any.
    pub fn arrow_canonical_for(&self, vortex_id: &ExtId) -> Option<ExtId> {
        self.arrow_canonical.load().get(vortex_id).copied()
    }

    /// Returns the Vortex extension id aliased to the given Arrow canonical name, if any.
    pub fn vortex_id_for_arrow_canonical(&self, arrow_name: &str) -> Option<ExtId> {
        self.arrow_canonical
            .load()
            .get(&ExtId::new(arrow_name))
            .copied()
    }
}

/// Extension trait for accessing the DType session.
pub trait DTypeSessionExt: SessionExt {
    /// Get the DType session.
    fn dtypes(&self) -> Ref<'_, DTypeSession>;
}

impl<S: SessionExt> DTypeSessionExt for S {
    fn dtypes(&self) -> Ref<'_, DTypeSession> {
        self.get::<DTypeSession>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_canonical_re_registration_is_clean() {
        let session = DTypeSession::default();
        let v = ExtId::new("vortex.test");

        session.register_arrow_canonical(v, "arrow.foo");
        assert_eq!(
            session.arrow_canonical_for(&v),
            Some(ExtId::new("arrow.foo"))
        );
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.foo"), Some(v));

        session.register_arrow_canonical(v, "arrow.bar");
        assert_eq!(
            session.arrow_canonical_for(&v),
            Some(ExtId::new("arrow.bar"))
        );
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.bar"), Some(v));
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.foo"), None);
    }

    /// `(a → b, b → a)` then `register(a, c)` should leave `(a → c, c → a)` only.
    #[test]
    fn rebind_vortex_id_to_new_arrow_name() {
        let session = DTypeSession::default();
        let a = ExtId::new("vortex.a");
        let b = ExtId::new("arrow.b");
        let c = ExtId::new("arrow.c");

        session.register_arrow_canonical(a, "arrow.b");
        assert_eq!(session.arrow_canonical_for(&a), Some(b));
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.b"), Some(a));

        session.register_arrow_canonical(a, "arrow.c");

        assert_eq!(session.arrow_canonical_for(&a), Some(c));
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.c"), Some(a));
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.b"), None);
    }

    /// `(a → b, b → a)` then `register(c, b)` should leave `(c → b, b → c)` only.
    #[test]
    fn steal_arrow_name_from_another_vortex_id() {
        let session = DTypeSession::default();
        let a = ExtId::new("vortex.a");
        let b = ExtId::new("arrow.b");
        let c = ExtId::new("vortex.c");

        session.register_arrow_canonical(a, "arrow.b");
        assert_eq!(session.arrow_canonical_for(&a), Some(b));

        session.register_arrow_canonical(c, "arrow.b");

        assert_eq!(session.arrow_canonical_for(&c), Some(b));
        assert_eq!(session.vortex_id_for_arrow_canonical("arrow.b"), Some(c));
        assert_eq!(session.arrow_canonical_for(&a), None);
    }
}
