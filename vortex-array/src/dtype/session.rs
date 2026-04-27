// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_error::VortexResult;
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

/// Converters between an extension's on-disk metadata bytes and the Arrow canonical JSON wire.
///
/// Bundled with the alias at registration time so [`ExtVTable`] stays Arrow-unaware.
#[derive(Copy, Clone, Debug)]
pub struct ArrowCanonicalCodec {
    pub to_json: fn(&[u8]) -> VortexResult<String>,
    pub from_json: fn(&str) -> VortexResult<Vec<u8>>,
}

#[derive(Copy, Clone, Debug)]
struct AliasEntry {
    /// Forward entries point at the Arrow canonical id; reverse entries point at the Vortex id.
    partner: ExtId,
    /// Same codec value in both directions of a registration; eviction relies on this.
    codec: ArrowCanonicalCodec,
}

#[derive(Debug, Default)]
struct ArrowCanonicalAliases(ArcSwap<HashMap<ExtId, AliasEntry>>);

impl ArrowCanonicalAliases {
    /// Re-registering evicts the previous mapping for either side so the bidirectional invariant
    /// holds after every call.
    fn register(&self, vortex_id: ExtId, arrow_name: &'static str, codec: ArrowCanonicalCodec) {
        let arrow_id = ExtId::new(arrow_name);
        let forward = AliasEntry {
            partner: arrow_id,
            codec,
        };
        let reverse = AliasEntry {
            partner: vortex_id,
            codec,
        };
        self.0.rcu(|prev| {
            let mut next = (**prev).clone();
            if let Some(stale) = next.insert(vortex_id, forward) {
                next.remove(&stale.partner);
            }
            if let Some(stale) = next.insert(arrow_id, reverse) {
                next.remove(&stale.partner);
            }
            Arc::new(next)
        });
    }

    fn arrow_canonical_for(&self, vortex_id: &ExtId) -> Option<AliasEntry> {
        self.0.load().get(vortex_id).copied()
    }

    fn vortex_id_for_arrow_canonical(&self, arrow_name: &str) -> Option<AliasEntry> {
        self.0.load().get(&ExtId::new(arrow_name)).copied()
    }
}

/// Session for managing extension dtypes.
#[derive(Debug)]
pub struct DTypeSession {
    registry: ExtDTypeRegistry,
    arrow_canonical: ArrowCanonicalAliases,
}

impl Default for DTypeSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
            arrow_canonical: ArrowCanonicalAliases::default(),
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

    /// Alias `arrow_name` to `vortex_id` with the codec used at the Arrow boundary.
    /// Re-registering evicts the previous mapping for either side.
    pub fn register_arrow_canonical(
        &self,
        vortex_id: ExtId,
        arrow_name: &'static str,
        codec: ArrowCanonicalCodec,
    ) {
        self.arrow_canonical.register(vortex_id, arrow_name, codec);
    }

    /// Returns the Arrow canonical name and codec aliased to `vortex_id`, if any.
    pub fn arrow_canonical_for(&self, vortex_id: &ExtId) -> Option<(ExtId, ArrowCanonicalCodec)> {
        self.arrow_canonical
            .arrow_canonical_for(vortex_id)
            .map(|e| (e.partner, e.codec))
    }

    /// Returns the Vortex id and codec aliased to `arrow_name`, if any.
    pub fn vortex_id_for_arrow_canonical(
        &self,
        arrow_name: &str,
    ) -> Option<(ExtId, ArrowCanonicalCodec)> {
        self.arrow_canonical
            .vortex_id_for_arrow_canonical(arrow_name)
            .map(|e| (e.partner, e.codec))
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
    use vortex_error::vortex_err;

    use super::*;

    const TEST_CODEC: ArrowCanonicalCodec = ArrowCanonicalCodec {
        to_json: |bytes| {
            String::from_utf8(bytes.to_vec()).map_err(|e| vortex_err!("non-utf8 test bytes: {e}"))
        },
        from_json: |s| Ok(s.as_bytes().to_vec()),
    };

    #[test]
    fn arrow_canonical_re_registration_is_clean() {
        let session = DTypeSession::default();
        let v = ExtId::new("vortex.test");

        session.register_arrow_canonical(v, "arrow.foo", TEST_CODEC);
        assert_eq!(
            session.arrow_canonical_for(&v).map(|(id, _)| id),
            Some(ExtId::new("arrow.foo"))
        );
        assert_eq!(
            session
                .vortex_id_for_arrow_canonical("arrow.foo")
                .map(|(id, _)| id),
            Some(v)
        );

        session.register_arrow_canonical(v, "arrow.bar", TEST_CODEC);
        assert_eq!(
            session.arrow_canonical_for(&v).map(|(id, _)| id),
            Some(ExtId::new("arrow.bar"))
        );
        assert_eq!(
            session
                .vortex_id_for_arrow_canonical("arrow.bar")
                .map(|(id, _)| id),
            Some(v)
        );
        assert!(session.vortex_id_for_arrow_canonical("arrow.foo").is_none());
    }

    /// `(vid → old, old → vid)` then `register(vid, new)` should leave `(vid → new, new → vid)`.
    #[test]
    fn rebind_vortex_id_to_new_arrow_name() {
        let session = DTypeSession::default();
        let vid = ExtId::new("vortex.a");
        let old = ExtId::new("arrow.b");
        let new = ExtId::new("arrow.c");

        session.register_arrow_canonical(vid, "arrow.b", TEST_CODEC);
        assert_eq!(
            session.arrow_canonical_for(&vid).map(|(id, _)| id),
            Some(old)
        );
        assert_eq!(
            session
                .vortex_id_for_arrow_canonical("arrow.b")
                .map(|(id, _)| id),
            Some(vid)
        );

        session.register_arrow_canonical(vid, "arrow.c", TEST_CODEC);

        assert_eq!(
            session.arrow_canonical_for(&vid).map(|(id, _)| id),
            Some(new)
        );
        assert_eq!(
            session
                .vortex_id_for_arrow_canonical("arrow.c")
                .map(|(id, _)| id),
            Some(vid)
        );
        assert!(session.vortex_id_for_arrow_canonical("arrow.b").is_none());
    }

    /// `(old → name, name → old)` then `register(new, name)` should leave `(new → name, name → new)`.
    #[test]
    fn steal_arrow_name_from_another_vortex_id() {
        let session = DTypeSession::default();
        let old = ExtId::new("vortex.a");
        let name = ExtId::new("arrow.b");
        let new = ExtId::new("vortex.c");

        session.register_arrow_canonical(old, "arrow.b", TEST_CODEC);
        assert_eq!(
            session.arrow_canonical_for(&old).map(|(id, _)| id),
            Some(name)
        );

        session.register_arrow_canonical(new, "arrow.b", TEST_CODEC);

        assert_eq!(
            session.arrow_canonical_for(&new).map(|(id, _)| id),
            Some(name)
        );
        assert_eq!(
            session
                .vortex_id_for_arrow_canonical("arrow.b")
                .map(|(id, _)| id),
            Some(new)
        );
        assert!(session.arrow_canonical_for(&old).is_none());
    }

    #[test]
    fn codec_round_trips_through_lookup() {
        let session = DTypeSession::default();
        let vid = ExtId::new("vortex.x");

        session.register_arrow_canonical(vid, "arrow.x", TEST_CODEC);

        let (_, codec) = session.arrow_canonical_for(&vid).unwrap();
        let json = (codec.to_json)(b"hello").unwrap();
        assert_eq!(json, "hello");
        let bytes = (codec.from_json)(&json).unwrap();
        assert_eq!(bytes, b"hello");
    }
}
