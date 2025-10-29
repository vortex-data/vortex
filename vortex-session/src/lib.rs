// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use dashmap::DashMap;
use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use vortex_error::VortexExpect;

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions, dtypes,
/// etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
#[derive(Debug)]
pub struct VortexSession(Arc<SessionVars>);

impl VortexSession {
    /// Creates an empty Vortex session.
    ///
    /// Do not call this function otherwise you will end up with an empty session!
    pub fn _empty() -> Self {
        Self(Arc::new(
            DashMap::with_hasher(BuildHasherDefault::default()),
        ))
    }

    /// Returns the scope variable of type `V` if it exists.
    pub fn get<V: SessionVar>(&self) -> Option<impl Deref<Target = V>> {
        self.0.get(&TypeId::of::<V>()).map(|entry| {
            entry.map(|v| {
                v.as_any()
                    .downcast_ref::<V>()
                    .vortex_expect("Type mismatch - this is a bug")
            })
        })
    }

    /// Returns the scope variable of type `V` if it exists.
    pub fn get_or_create<V: SessionVar + Default>(&self) -> impl Deref<Target = V> {
        self.0
            .entry(TypeId::of::<V>())
            .or_insert_with(|| Arc::new(V::default()))
            .value()
            .clone()
            .as_any_arc()
            .downcast::<V>()
            .ok()
            .vortex_expect("Type mismatch - this is a bug")
    }
}

/// A TypeMap based on `https://docs.rs/http/1.2.0/src/http/extensions.rs.html#41-266`.
type SessionVars = DashMap<TypeId, Arc<dyn SessionVar>, BuildHasherDefault<IdHasher>>;

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
pub trait SessionVar: Any + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

impl<T: Send + Sync + Debug + 'static> SessionVar for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>
    where
        Self: Sized,
    {
        self
    }
}
