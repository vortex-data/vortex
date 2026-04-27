// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod registry;

use std::any::Any;
use std::any::TypeId;
use std::any::type_name;
use std::fmt::Debug;
use std::hash::BuildHasherDefault;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use dashmap::DashMap;
use dashmap::Entry;
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
        match self.0.entry(TypeId::of::<V>()) {
            Entry::Occupied(_) => {
                vortex_panic!(
                    "Session variable of type {} already exists",
                    type_name::<V>()
                );
            }
            Entry::Vacant(e) => {
                e.insert(Box::new(var));
            }
        }
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

/// Trait for accessing and modifying the state of a Vortex session.
pub trait SessionExt: Sized + private::Sealed {
    /// Returns the [`VortexSession`].
    fn session(&self) -> VortexSession;

    /// Returns the scope variable of type `V`, or inserts a default one if it does not exist.
    fn get<V: SessionVar + Default>(&self) -> Ref<'_, V>;

    /// Returns the scope variable of type `V` if it exists.
    fn get_opt<V: SessionVar>(&self) -> Option<Ref<'_, V>>;

    /// Returns the scope variable of type `V`, or inserts a default one if it does not exist.
    ///
    /// Note that the returned value internally holds a lock on the variable.
    fn get_mut<V: SessionVar + Default>(&self) -> RefMut<'_, V>;

    /// Returns the scope variable of type `V`, if it exists.
    ///
    /// Note that the returned value internally holds a lock on the variable.
    fn get_mut_opt<V: SessionVar>(&self) -> Option<RefMut<'_, V>>;
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
        // NOTE(ngates): we don't use `entry().or_insert_with_key()` here because the DashMap
        //  would immediately acquire an exclusive write lock.
        if let Some(v) = self.0.get(&TypeId::of::<V>()) {
            return Ref(v.map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("Type mismatch - this is a bug")
            }));
        }

        // If we get here, the value was not present, so we insert the default with a write lock.
        Ref(self
            .0
            .entry(TypeId::of::<V>())
            .or_insert_with(|| Box::new(V::default()))
            .downgrade()
            .map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("Type mismatch - this is a bug")
            }))
    }

    fn get_opt<V: SessionVar>(&self) -> Option<Ref<'_, V>> {
        self.0.get(&TypeId::of::<V>()).map(|v| {
            Ref(v.map(|v| {
                (**v)
                    .as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("Type mismatch - this is a bug")
            }))
        })
    }

    /// Returns the scope variable of type `V`, or inserts a default one if it does not exist.
    ///
    /// Note that the returned value internally holds a lock on the variable.
    fn get_mut<V: SessionVar + Default>(&self) -> RefMut<'_, V> {
        RefMut(
            self.0
                .entry(TypeId::of::<V>())
                .or_insert_with(|| Box::new(V::default()))
                .map(|v| {
                    (**v)
                        .as_any_mut()
                        .downcast_mut::<V>()
                        .vortex_expect("Type mismatch - this is a bug")
                }),
        )
    }

    fn get_mut_opt<V: SessionVar>(&self) -> Option<RefMut<'_, V>> {
        self.0.get_mut(&TypeId::of::<V>()).map(|v| {
            RefMut(v.map(|v| {
                (**v)
                    .as_any_mut()
                    .downcast_mut::<V>()
                    .vortex_expect("Type mismatch - this is a bug")
            }))
        })
    }
}

/// A TypeMap based on `https://docs.rs/http/1.2.0/src/http/extensions.rs.html#41-266`.
type SessionVars = DashMap<TypeId, Box<dyn SessionVar>, BuildHasherDefault<IdHasher>>;

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
pub trait SessionVar: Any + Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Send + Sync + Debug + 'static> SessionVar for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// NOTE(ngates): we don't want to expose that the internals of a session is a DashMap, so we have
// our own wrapped Ref type.
pub struct Ref<'a, T>(dashmap::mapref::one::MappedRef<'a, TypeId, Box<dyn SessionVar>, T>);
impl<'a, T> Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a, T> Ref<'a, T> {
    /// Map this reference to a different target.
    pub fn map<F, U>(self, f: F) -> Ref<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        Ref(self.0.map(f))
    }
}

pub struct RefMut<'a, T>(dashmap::mapref::one::MappedRefMut<'a, TypeId, Box<dyn SessionVar>, T>);
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
    /// Map this mutable reference to a different target.
    pub fn map<F, U>(self, f: F) -> RefMut<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        RefMut(self.0.map(f))
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
