// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScanCtx`] — per-scan execution context.
//!
//! A typed key/value map (mirroring [`vortex_session::VortexSession`])
//! threaded through every [`crate::v2::plan::LayoutPlan::execute`]
//! call so plan nodes can stash arbitrary per-scan state without
//! holding caches on the plan struct itself. Plans must remain pure
//! descriptions — see `LAYOUT_PLAN.md` § Model.
//!
//! Each variable type has at most one slot per `ScanCtx`. Values that
//! need to fan out by some sub-key (e.g., the [`crate::v2::let_use`]
//! `LetId`-keyed registry) wrap an internal map under one type slot.
//!
//! See `LAYOUT_PLAN.md` § Tee and CommonSubplanElimination for how
//! `Let` / `Use` use this map; future plan nodes can layer their own
//! [`ScanCtxValue`] types beside it.

use std::any::Any;
use std::any::TypeId;
use std::fmt::Debug;
use std::hash::BuildHasherDefault;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use dashmap::DashMap;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

/// Per-scan execution context. Cheap to clone (single [`Arc`]).
///
/// Carries the [`VortexSession`] used at execute time and a typed
/// key/value map for per-scan state that plan nodes need to share
/// across `execute` calls. Constructed once by the engine when a scan
/// begins, then passed by reference into every
/// [`crate::v2::plan::LayoutPlan::execute`] call for that scan.
#[derive(Clone, Debug)]
pub struct ScanCtx {
    session: VortexSession,
    vars: Arc<ScanVars>,
}

impl ScanCtx {
    /// Construct a context bound to the given session.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            vars: Arc::default(),
        }
    }

    /// Construct a context bound to an empty session. Mostly useful
    /// for tests; production callers should pass a real session.
    pub fn empty() -> Self {
        Self::new(VortexSession::empty())
    }

    /// The session this scan is executing against.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Get a typed slot, inserting a default value if it is not yet
    /// present. The returned [`Ref`] holds a read lock on the slot
    /// for its lifetime.
    pub fn get<V: ScanCtxValue + Default>(&self) -> Ref<'_, V> {
        // Same shape as VortexSession::get — try a read first to avoid
        // grabbing a write lock when the value already exists.
        if let Some(v) = self.vars.get(&TypeId::of::<V>()) {
            return Ref(v.map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("ScanCtx type mismatch — bug")
            }));
        }
        Ref(self
            .vars
            .entry(TypeId::of::<V>())
            .or_insert_with(|| Box::new(V::default()))
            .downgrade()
            .map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("ScanCtx type mismatch — bug")
            }))
    }

    /// Get a typed slot if it has been initialised. Returns `None`
    /// when the slot has never been touched.
    pub fn get_opt<V: ScanCtxValue>(&self) -> Option<Ref<'_, V>> {
        self.vars.get(&TypeId::of::<V>()).map(|v| {
            Ref(v.map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("ScanCtx type mismatch — bug")
            }))
        })
    }

    /// Get a mutable typed slot, inserting a default value if it is
    /// not yet present. The returned [`RefMut`] holds a write lock on
    /// the slot for its lifetime.
    pub fn get_mut<V: ScanCtxValue + Default>(&self) -> RefMut<'_, V> {
        RefMut(
            self.vars
                .entry(TypeId::of::<V>())
                .or_insert_with(|| Box::new(V::default()))
                .map(|v| {
                    (**v)
                        .as_any_mut()
                        .downcast_mut::<V>()
                        .vortex_expect("ScanCtx type mismatch — bug")
                }),
        )
    }
}

/// A value stored in a [`ScanCtx`]. Implementors get a single typed
/// slot per `ScanCtx`. Use a wrapper type holding an internal map if
/// you need to fan out by a sub-key (see
/// [`crate::v2::let_use::LetRegistry`]).
pub trait ScanCtxValue: Any + Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

type ScanVars = DashMap<TypeId, Box<dyn ScanCtxValue>, BuildHasherDefault<IdHasher>>;

/// `TypeId`s already are hashes, so the hasher is a no-op identity.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("TypeId calls write_u64");
    }

    #[inline]
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }
}

/// A read guard on a typed slot. Holds a read lock on the underlying
/// map shard for its lifetime.
pub struct Ref<'a, T>(dashmap::mapref::one::MappedRef<'a, TypeId, Box<dyn ScanCtxValue>, T>);

impl<'a, T> Deref for Ref<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> Ref<'a, T> {
    pub fn map<F, U>(self, f: F) -> Ref<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        Ref(self.0.map(f))
    }
}

/// A write guard on a typed slot. Holds a write lock on the underlying
/// map shard for its lifetime.
pub struct RefMut<'a, T>(dashmap::mapref::one::MappedRefMut<'a, TypeId, Box<dyn ScanCtxValue>, T>);

impl<'a, T> Deref for RefMut<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> DerefMut for RefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<'a, T> RefMut<'a, T> {
    pub fn map<F, U>(self, f: F) -> RefMut<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        RefMut(self.0.map(f))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use std::any::Any;
    use std::fmt::Debug;

    use super::ScanCtx;
    use super::ScanCtxValue;

    #[derive(Debug, Default)]
    struct Counter(usize);

    impl ScanCtxValue for Counter {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[derive(Debug, Default)]
    struct OtherCounter(usize);

    impl ScanCtxValue for OtherCounter {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Counter {
        fn value(&self) -> usize {
            self.0
        }
        fn set(&mut self, v: usize) {
            self.0 = v;
        }
    }

    impl OtherCounter {
        fn value(&self) -> usize {
            self.0
        }
        fn set(&mut self, v: usize) {
            self.0 = v;
        }
    }

    #[test]
    fn get_default_inits_once() {
        let ctx = ScanCtx::empty();
        assert_eq!(ctx.get::<Counter>().value(), 0);
        ctx.get_mut::<Counter>().set(7);
        assert_eq!(ctx.get::<Counter>().value(), 7);
    }

    #[test]
    fn get_opt_returns_none_until_initialised() {
        let ctx = ScanCtx::empty();
        assert!(ctx.get_opt::<Counter>().is_none());
        drop(ctx.get::<Counter>());
        assert!(ctx.get_opt::<Counter>().is_some());
    }

    #[test]
    fn distinct_types_get_distinct_slots() {
        let ctx = ScanCtx::empty();
        ctx.get_mut::<Counter>().set(1);
        ctx.get_mut::<OtherCounter>().set(2);
        assert_eq!(ctx.get::<Counter>().value(), 1);
        assert_eq!(ctx.get::<OtherCounter>().value(), 2);
    }
}
