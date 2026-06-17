// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod immutable;
pub mod registry;

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hasher;

pub use immutable::SessionBuilder;
pub use immutable::VortexSession;
pub use immutable::VortexSessionVar;

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

/// Trait for accessing the state of a Vortex session.
pub trait SessionExt: Sized + private::Sealed {
    /// Returns the [`VortexSession`].
    fn session(&self) -> VortexSession;

    /// Returns the session variable of type `V`.
    ///
    /// # Panics
    ///
    /// If no variable of that type exists in this session.
    fn get<V: VortexSessionVar>(&self) -> &V;

    /// Returns the session variable of type `V` if it exists.
    fn get_opt<V: VortexSessionVar>(&self) -> Option<&V>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::VortexSession {}
}

impl SessionExt for VortexSession {
    fn session(&self) -> VortexSession {
        self.clone()
    }

    fn get<V: VortexSessionVar>(&self) -> &V {
        VortexSession::get::<V>(self)
    }

    fn get_opt<V: VortexSessionVar>(&self) -> Option<&V> {
        VortexSession::get_opt::<V>(self)
    }
}

/// This trait defines variables that can be stored against a Vortex session.
///
/// Users should implement this trait for anything that you want to store on a `VortexSession`.
pub trait SessionVar: Any + Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

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
