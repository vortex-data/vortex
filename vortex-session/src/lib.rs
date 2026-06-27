// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod registry;
mod session;

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hasher;

pub use session::SessionGuard;
pub use session::SessionMut;
pub use session::VortexSession;
pub use session::VortexSessionVar;

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

    /// Returns the session variable of type `V`, inserting a default one if it does not exist.
    ///
    /// The default is constructed and inserted copy-on-write: `V::default()` runs without any lock
    /// held, so it may freely re-enter the session, and a concurrent insert of the same type is
    /// resolved by keeping the first value published.
    fn get<V: VortexSessionVar + Default>(&self) -> SessionGuard<'_, V>;

    /// Returns the session variable of type `V` if it exists.
    fn get_opt<V: VortexSessionVar>(&self) -> Option<SessionGuard<'_, V>>;

    /// Returns a copy-on-write [`SessionMut`] handle for the variable of type `V`, inserting a
    /// default one first if it does not exist.
    ///
    /// The handle starts as a clone of the current value; mutating it through `DerefMut` and
    /// dropping it publishes the result back into the session copy-on-write — the ergonomic
    /// equivalent of reading the variable, modifying a clone, and re-registering it with
    /// [`VortexSession::register`].
    fn get_mut<V: VortexSessionVar + Default + Clone>(&self) -> SessionMut<'_, V>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::VortexSession {}
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
