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

/// Forward map is the canonical source: each Vortex extension owns its codec and points at the
/// Arrow canonical name it serializes as. Reverse map is a lookup index for the read path,
/// taking an Arrow name back to the Vortex id whose codec applies.
#[derive(Default, Clone)]
struct AliasState {
    forward: HashMap<ExtId, (ExtId, ArrowCanonicalCodec)>,
    reverse: HashMap<ExtId, ExtId>,
}

#[derive(Debug, Default)]
struct ArrowCanonicalAliases(ArcSwap<AliasState>);

impl ArrowCanonicalAliases {
    /// Re-registering evicts any prior alias touching either id so both directions agree.
    fn register(&self, vortex_id: ExtId, arrow_id: ExtId, codec: ArrowCanonicalCodec) {
        self.0.rcu(|prev| {
            let mut next = (**prev).clone();
            if let Some((stale_arrow, _)) = next.forward.remove(&vortex_id) {
                next.reverse.remove(&stale_arrow);
            }
            if let Some(stale_vortex) = next.reverse.remove(&arrow_id) {
                next.forward.remove(&stale_vortex);
            }
            next.forward.insert(vortex_id, (arrow_id, codec));
            next.reverse.insert(arrow_id, vortex_id);
            Arc::new(next)
        });
    }

    fn arrow_alias_for(&self, vortex_id: &ExtId) -> Option<(ExtId, ArrowCanonicalCodec)> {
        self.0.load().forward.get(vortex_id).copied()
    }

    fn vortex_alias_for(&self, arrow_id: &ExtId) -> Option<(ExtId, ArrowCanonicalCodec)> {
        let state = self.0.load();
        let vortex_id = *state.reverse.get(arrow_id)?;
        let (_, codec) = *state.forward.get(&vortex_id)?;
        Some((vortex_id, codec))
    }
}

impl std::fmt::Debug for AliasState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AliasState")
            .field("forward", &self.forward)
            .field("reverse", &self.reverse)
            .finish()
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

    /// Alias `arrow_id` to `vortex_id` with the codec used at the Arrow boundary.
    /// Re-registering evicts the previous mapping for either side.
    pub fn register_arrow_canonical(
        &self,
        vortex_id: ExtId,
        arrow_id: ExtId,
        codec: ArrowCanonicalCodec,
    ) {
        self.arrow_canonical.register(vortex_id, arrow_id, codec);
    }

    /// Returns the Arrow canonical id and codec aliased to `vortex_id`, if any.
    pub fn arrow_alias_for(&self, vortex_id: &ExtId) -> Option<(ExtId, ArrowCanonicalCodec)> {
        self.arrow_canonical.arrow_alias_for(vortex_id)
    }

    /// Returns the Vortex id and codec aliased to `arrow_id`, if any.
    pub fn vortex_alias_for(&self, arrow_id: &ExtId) -> Option<(ExtId, ArrowCanonicalCodec)> {
        self.arrow_canonical.vortex_alias_for(arrow_id)
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
        let foo = ExtId::new("arrow.foo");
        let bar = ExtId::new("arrow.bar");

        session.register_arrow_canonical(v, foo, TEST_CODEC);
        assert_eq!(session.arrow_alias_for(&v).map(|(id, _)| id), Some(foo));
        assert_eq!(session.vortex_alias_for(&foo).map(|(id, _)| id), Some(v));

        session.register_arrow_canonical(v, bar, TEST_CODEC);
        assert_eq!(session.arrow_alias_for(&v).map(|(id, _)| id), Some(bar));
        assert_eq!(session.vortex_alias_for(&bar).map(|(id, _)| id), Some(v));
        assert!(session.vortex_alias_for(&foo).is_none());
    }

    /// `(vid → old, old → vid)` then `register(vid, new)` should leave `(vid → new, new → vid)`.
    #[test]
    fn rebind_vortex_id_to_new_arrow_name() {
        let session = DTypeSession::default();
        let vid = ExtId::new("vortex.a");
        let old = ExtId::new("arrow.b");
        let new = ExtId::new("arrow.c");

        session.register_arrow_canonical(vid, old, TEST_CODEC);
        assert_eq!(session.arrow_alias_for(&vid).map(|(id, _)| id), Some(old));
        assert_eq!(session.vortex_alias_for(&old).map(|(id, _)| id), Some(vid));

        session.register_arrow_canonical(vid, new, TEST_CODEC);

        assert_eq!(session.arrow_alias_for(&vid).map(|(id, _)| id), Some(new));
        assert_eq!(session.vortex_alias_for(&new).map(|(id, _)| id), Some(vid));
        assert!(session.vortex_alias_for(&old).is_none());
    }

    /// `(old → name, name → old)` then `register(new, name)` should leave `(new → name, name → new)`.
    #[test]
    fn steal_arrow_name_from_another_vortex_id() {
        let session = DTypeSession::default();
        let old = ExtId::new("vortex.a");
        let name = ExtId::new("arrow.b");
        let new = ExtId::new("vortex.c");

        session.register_arrow_canonical(old, name, TEST_CODEC);
        assert_eq!(session.arrow_alias_for(&old).map(|(id, _)| id), Some(name));

        session.register_arrow_canonical(new, name, TEST_CODEC);

        assert_eq!(session.arrow_alias_for(&new).map(|(id, _)| id), Some(name));
        assert_eq!(session.vortex_alias_for(&name).map(|(id, _)| id), Some(new));
        assert!(session.arrow_alias_for(&old).is_none());
    }

    #[test]
    fn codec_round_trips_through_lookup() {
        let session = DTypeSession::default();
        let vid = ExtId::new("vortex.x");
        let aid = ExtId::new("arrow.x");

        session.register_arrow_canonical(vid, aid, TEST_CODEC);

        let (_, codec) = session.arrow_alias_for(&vid).unwrap();
        let json = (codec.to_json)(b"hello").unwrap();
        assert_eq!(json, "hello");
        let bytes = (codec.from_json)(&json).unwrap();
        assert_eq!(bytes, b"hello");
    }
}
