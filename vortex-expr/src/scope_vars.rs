// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::{Any, TypeId};
use std::hash::{BuildHasherDefault, Hasher};

use vortex_utils::aliases::hash_map::HashMap;

/// A TypeMap based on `https://docs.rs/http/1.2.0/src/http/extensions.rs.html#41-266`.
pub(crate) type ScopeVars = HashMap<TypeId, Box<dyn ScopeVar>, BuildHasherDefault<IdHasher>>;

/// With TypeIds as keys, there's no need to hash them. They are already hashes
/// themselves, coming from the compiler. The IdHasher just holds the u64 of
/// the TypeId, and then returns it, instead of doing any bit fiddling.
#[derive(Default)]
pub(super) struct IdHasher(u64);

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

/// A trait for scope variables that can be stored in a `ScopeVars` map.
pub trait ScopeVar: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn ScopeVar>;
}

impl Clone for Box<dyn ScopeVar> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

impl<T: Clone + Send + Sync + 'static> ScopeVar for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ScopeVar> {
        Box::new(self.clone())
    }
}
