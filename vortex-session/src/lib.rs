// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod registry;

use std::any::Any;
use std::any::TypeId;
use std::any::type_name;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::BuildHasherDefault;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions, dtypes,
/// etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
#[derive(Clone, Debug)]
pub struct VortexSession(Arc<SessionVars>);

impl VortexSession {
    /// Create a new [`VortexSession`] with no session state.
    ///
    /// It is recommended to use the `default()` method instead provided by the main `vortex` crate.
    pub fn empty() -> Self {
        Self(Default::default())
    }

    /// Inserts a new session variable of type `V` with its default value.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with<V: SessionVar + Default>(self) -> Self {
        self.with_some(V::default())
    }

    /// Inserts a new session variable of type `V`.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with_some<V: SessionVar>(self, var: V) -> Self {
        let tid = TypeId::of::<V>();
        // Wrap once: the rcu closure may run multiple times under contention.
        let arc: Arc<dyn SessionVar> = Arc::new(var);
        self.0.rcu(|map| {
            if map.contains_key(&tid) {
                vortex_panic!(
                    "Session variable of type {} already exists",
                    type_name::<V>()
                );
            }
            let mut new = HashMap::clone(map);
            new.insert(tid, arc.clone());
            new
        });
        self
    }

    /// Allow deserializing unknown plugin IDs as non-executable foreign placeholders.
    pub fn allow_unknown(self) -> Self {
        let mut policy = <Self as SessionExt>::get_mut::<UnknownPluginPolicy>(&self);
        policy.allow_unknown = true;
        drop(policy);
        self
    }

    /// Returns whether unknown plugins should deserialize as foreign placeholders.
    pub fn allows_unknown(&self) -> bool {
        <Self as SessionExt>::get_opt::<UnknownPluginPolicy>(self)
            .map(|p| p.allow_unknown)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct UnknownPluginPolicy {
    allow_unknown: bool,
}

impl SessionVar for UnknownPluginPolicy {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Trait for accessing and modifying the state of a Vortex session.
pub trait SessionExt: Sized + private::Sealed {
    /// Returns the [`VortexSession`].
    fn session(&self) -> VortexSession;

    /// Returns the scope variable of type `V`, or inserts a default one if it does not exist.
    fn get<V: SessionVar + Default>(&self) -> Ref<'_, V>;

    /// Returns the scope variable of type `V` if it exists.
    fn get_opt<V: SessionVar>(&self) -> Option<Ref<'_, V>>;

    /// Returns mutable access to the scope variable of type `V`, inserting a default if it does
    /// not exist.
    ///
    /// The store keeps variables behind shared snapshots, so this returns a *copy* that is
    /// written back into the session when the [`RefMut`] is dropped (read-copy-update). The
    /// mutation is therefore not observable through other handles until the returned guard drops —
    /// fine for the setup-time builders that use it. Hence the `Clone` bound.
    fn get_mut<V: SessionVar + Default + Clone>(&self) -> RefMut<'_, V>;

    /// Like [`get_mut`](Self::get_mut), but returns `None` if the variable does not exist.
    fn get_mut_opt<V: SessionVar + Clone>(&self) -> Option<RefMut<'_, V>>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::VortexSession {}
}

impl SessionExt for VortexSession {
    fn session(&self) -> VortexSession {
        self.clone()
    }

    /// Returns the scope variable of type `V`, or inserts a default one if it does not exist.
    fn get<V: SessionVar + Default>(&self) -> Ref<'_, V> {
        if let Some(r) = self.get_opt::<V>() {
            return r;
        }
        // Not present: insert a default via a copy-on-write swap, then read it back. The rcu may
        // race with other inserters; whoever wins, the value is present afterwards.
        let tid = TypeId::of::<V>();
        let arc: Arc<dyn SessionVar> = Arc::new(V::default());
        self.0.rcu(|map| {
            let mut new = HashMap::clone(map);
            new.entry(tid).or_insert_with(|| arc.clone());
            new
        });
        self.get_opt::<V>()
            .vortex_expect("default was just inserted")
    }

    fn get_opt<V: SessionVar>(&self) -> Option<Ref<'_, V>> {
        // Lock-free read: load the current map snapshot (a plain atomic load — no shard RwLock),
        // clone the value's Arc so it outlives the load guard, and downcast.
        let map = self.0.load();
        let arc = map.get(&TypeId::of::<V>())?.clone();
        let ptr = (*arc)
            .as_any()
            .downcast_ref::<V>()
            .vortex_expect("Type mismatch - this is a bug") as *const V;
        Some(Ref {
            _owner: arc,
            ptr,
            _marker: PhantomData,
        })
    }

    fn get_mut<V: SessionVar + Default + Clone>(&self) -> RefMut<'_, V> {
        // Read-copy-update: hand back a copy, written back into the store on drop.
        let value = self.get::<V>().clone();
        RefMut {
            session: self.session(),
            value,
            _marker: PhantomData,
        }
    }

    fn get_mut_opt<V: SessionVar + Clone>(&self) -> Option<RefMut<'_, V>> {
        let value = self.get_opt::<V>()?.clone();
        Some(RefMut {
            session: self.session(),
            value,
            _marker: PhantomData,
        })
    }
}

/// A read-lock-free typemap: writes (rare, setup-time) copy-on-write swap the whole map; reads
/// are plain atomic loads of the current snapshot, so concurrent per-node lookups never contend
/// a shard lock the way a `DashMap` keyed by a constant `TypeId` does.
type SessionVars = ArcSwap<HashMap<TypeId, Arc<dyn SessionVar>, BuildHasherDefault<IdHasher>>>;

/// With TypeIds as keys, there's no need to hash them. They are already hashes
/// themselves, coming from the compiler. The IdHasher just holds the u64 of
/// the TypeId, and then returns it, instead of doing any bit fiddling.
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

/// This trait defines variables that can be stored against a Vortex session.
///
/// Users should implement this trait for anything that you want to store on a `VortexSession`.
pub trait SessionVar: Any + Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// An owned, session-var-lifetime reference. Holds the value's `Arc` alive and a pointer into it,
// so it outlives the (temporary) `ArcSwap` load guard. The `'a` is vestigial (kept for API
// compatibility); the `Arc` is what actually keeps the borrow valid.
pub struct Ref<'a, T: ?Sized> {
    _owner: Arc<dyn SessionVar>,
    ptr: *const T,
    _marker: PhantomData<&'a T>,
}
// SAFETY: `ptr` points into `_owner`, a `Send + Sync` `SessionVar`; we only expose `&T`, so this
// is sound whenever `T: Sync` (shared `&T` across threads) and `T: Send` is not required.
unsafe impl<T: ?Sized + Sync> Send for Ref<'_, T> {}
unsafe impl<T: ?Sized + Sync> Sync for Ref<'_, T> {}
impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` points into `_owner`, which is kept alive for `self`'s lifetime.
        unsafe { &*self.ptr }
    }
}
impl<'a, T> Ref<'a, T> {
    /// Map this reference to a different target within the same owning value.
    pub fn map<F, U>(self, f: F) -> Ref<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        // SAFETY: `ptr` is valid for the borrow passed to `f`; the result points into `_owner`.
        let ptr = f(unsafe { &*self.ptr }) as *const U;
        Ref {
            _owner: self._owner,
            ptr,
            _marker: PhantomData,
        }
    }
}

/// A mutable handle to a session variable.
///
/// Because the store keeps variables behind shared, copy-on-write snapshots, this owns a working
/// copy of the variable and writes it back into the session on drop. Mutations are not visible to
/// other handles until this guard is dropped.
pub struct RefMut<'a, V: SessionVar + Clone> {
    session: VortexSession,
    value: V,
    _marker: PhantomData<&'a mut V>,
}
impl<V: SessionVar + Clone> Deref for RefMut<'_, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl<V: SessionVar + Clone> DerefMut for RefMut<'_, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}
impl<V: SessionVar + Clone> Drop for RefMut<'_, V> {
    fn drop(&mut self) {
        let tid = TypeId::of::<V>();
        let arc: Arc<dyn SessionVar> = Arc::new(self.value.clone());
        self.session.0.rcu(|map| {
            let mut new = HashMap::clone(map);
            new.insert(tid, arc.clone());
            new
        });
    }
}

#[cfg(test)]
mod tests {
    use super::VortexSession;

    #[test]
    fn allow_unknown_flag_is_opt_in() {
        let session = VortexSession::empty();
        assert!(!session.allows_unknown());

        let session = session.allow_unknown();
        assert!(session.allows_unknown());
    }
}
