// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`VortexSession`] container.
//!
//! A [`VortexSession`] is a type-map of [`VortexSessionVar`]s keyed by [`TypeId`]. It is backed by
//! an [`ArcSwap`], giving lock-free reads and copy-on-write writes:
//!
//! * **Reads** ([`SessionExt::get`], [`SessionExt::get_opt`]) load the current snapshot
//!   without taking any lock and hand back a [`SessionGuard`] that derefs to the variable. Because a
//!   read never takes a lock, it can never deadlock, and because it never holds a lock across the
//!   returned reference there is no reader/writer contention.
//!
//! * **Writes** ([`VortexSession::register`], [`SessionExt::get`] on a missing default) are
//!   copy-on-write: the map is cloned, the change applied to the private copy, and the new map
//!   atomically published via [`ArcSwap::rcu`]. The closure passed to `rcu` only clones the map and
//!   inserts an already-constructed value, so no user code (in particular, no `Default::default`
//!   implementation) ever runs while a lock is held. This is the key difference from the previous
//!   `DashMap`-backed session, where `entry().or_insert_with(f)` ran `f` while holding the shard's
//!   write lock and could deadlock if `f` re-entered the session.
//!
//! A modified session is produced by mutating it **in place**: [`VortexSession::register`] — and
//! the configuration `with_*` helpers built on it — apply their change copy-on-write to the shared
//! backing cell. Clones of a session share that cell, so a variable registered through one clone
//! (or one `with_*` call) is visible to all of them. This is what late plugin/encoding registration
//! relies on.
//!
//! To build a session from scratch, start from [`VortexSession::empty`] and chain the `with_*`
//! helpers. Each [`empty`](VortexSession::empty) creates its own backing cell, so a session built
//! this way is independent of any other.

use std::any::TypeId;
use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use arc_swap::ArcSwap;
use arc_swap::Guard;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

use crate::IdHasher;
use crate::SessionExt;
use crate::SessionVar;
use crate::UnknownPluginPolicy;

/// A [`SessionVar`] that can be stored in a [`VortexSession`].
///
/// This trait is implemented automatically for every [`SessionVar`] that is also [`Clone`], so
/// types opt in by implementing [`Clone`] rather than implementing this trait directly. The trait
/// itself stays object-safe (so it can be stored as `Arc<dyn VortexSessionVar>`); the [`Clone`]
/// bound lives on the blanket impl rather than the trait.
///
/// Requiring [`Clone`] lets the configuration `with_*` helpers read a variable, modify a copy, and
/// re-[`register`](VortexSession::register) the result.
pub trait VortexSessionVar: SessionVar {}

impl<V: SessionVar + Clone> VortexSessionVar for V {}

/// The immutable type-map backing a published [`VortexSession`] snapshot.
type SessionVars = HashMap<TypeId, Arc<dyn VortexSessionVar>, BuildHasherDefault<IdHasher>>;

/// A reference to a session variable of type `V`, returned by [`SessionExt::get`] and
/// [`SessionExt::get_opt`].
///
/// It borrows the session's current snapshot through an [`arc_swap::Guard`], so reads never take a
/// lock or a full [`Arc`] clone. The guard is tied to the session borrow it was read from, so it is
/// meant to be used on the stack rather than stored in a long-lived data structure (holding it pins
/// an internal arc-swap slot, which can contend with concurrent writers). `SessionGuard` derefs to
/// `V`, so it can be used wherever a `&V` is expected.
pub struct SessionGuard<'a, V> {
    snapshot: Guard<Arc<SessionVars>>,
    _session: PhantomData<&'a VortexSession>,
    _marker: PhantomData<fn() -> V>,
}

impl<V: VortexSessionVar> Deref for SessionGuard<'_, V> {
    type Target = V;

    fn deref(&self) -> &V {
        // The constructor of `SessionGuard` guarantees the variable is present in `snapshot`.
        self.snapshot
            .get(&TypeId::of::<V>())
            .vortex_expect("SessionGuard invariant: variable present in snapshot")
            .as_any()
            .downcast_ref::<V>()
            .vortex_expect("Type mismatch - this is a bug")
    }
}

impl<V: VortexSessionVar> Debug for SessionGuard<'_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

/// A copy-on-write mutable handle to a session variable of type `V`, returned by
/// [`SessionExt::get_mut`].
///
/// It holds a private clone of the variable and exposes it through [`DerefMut`]. When the handle is
/// dropped, the (possibly mutated) value is re-[`register`](VortexSession::register)ed into the
/// session, replacing the previous one. Because the session stores each variable behind a shared
/// [`Arc`] observed by every clone, handing out a plain `&mut` to the stored value would be unsound;
/// this guard provides mutable access by cloning on read and publishing on drop instead.
pub struct SessionMut<'a, V: VortexSessionVar> {
    session: &'a VortexSession,
    // `Some` for the whole lifetime of the guard; taken in `drop` to move it back into the session.
    value: Option<V>,
}

impl<V: VortexSessionVar> Deref for SessionMut<'_, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value
            .as_ref()
            .vortex_expect("SessionMut invariant: value present until drop")
    }
}

impl<V: VortexSessionVar> DerefMut for SessionMut<'_, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value
            .as_mut()
            .vortex_expect("SessionMut invariant: value present until drop")
    }
}

impl<V: VortexSessionVar> Drop for SessionMut<'_, V> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            self.session.register(value);
        }
    }
}

impl<V: VortexSessionVar> Debug for SessionMut<'_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions,
/// dtypes, etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
///
/// Cloning a session is cheap and shares the backing store: a variable registered through one
/// clone (via [`VortexSession::register`], or one of the `with_*` helpers) is observed by all
/// clones. To build an *independent* session, start from [`VortexSession::empty`].
#[derive(Clone)]
pub struct VortexSession(Arc<ArcSwap<SessionVars>>);

impl Debug for VortexSession {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VortexSession")
            .field(&self.0.load().as_ref())
            .finish()
    }
}

impl VortexSession {
    /// Create a new [`VortexSession`] with no session state.
    pub fn empty() -> Self {
        Self(Arc::new(ArcSwap::from_pointee(SessionVars::default())))
    }

    /// Inserts `V::default()` if no variable of type `V` is present yet, copy-on-write.
    ///
    /// The default is constructed *outside* the [`ArcSwap::rcu`] closure, so `V::default()` never
    /// runs under a lock and is never run more than once — it may therefore freely re-enter the
    /// session (read or even register other variables) without risk of deadlock. The closure only
    /// clones the map and inserts the already-built value, so if a concurrent writer published a
    /// value (or the closure is retried under contention) the first-published value is kept and the
    /// prebuilt default is simply dropped.
    fn insert_default<V: VortexSessionVar + Default>(&self) {
        let default: Arc<dyn VortexSessionVar> = Arc::new(V::default());
        self.0.rcu(|current| {
            let mut next = SessionVars::clone(current);
            next.entry(TypeId::of::<V>())
                .or_insert_with(|| Arc::clone(&default));
            next
        });
    }

    /// Register a session variable of type `V`, replacing any existing variable of that type.
    ///
    /// The mutation is applied in place to the shared backing store, so it is visible through every
    /// clone of this session.
    pub fn register<V: VortexSessionVar>(&self, var: V) {
        let var: Arc<dyn VortexSessionVar> = Arc::new(var);
        self.0.rcu(|current| {
            let mut next = SessionVars::clone(current);
            next.insert(TypeId::of::<V>(), Arc::clone(&var));
            next
        });
    }

    /// Inserts a new session variable of type `V` with its default value, mutating this session in
    /// place and returning it for chaining.
    ///
    /// Like [`register`](Self::register), the change is applied to the shared backing store, so it
    /// is observed through every clone of this session.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with<V: VortexSessionVar + Default>(self) -> Self {
        self.with_some(V::default())
    }

    /// Inserts a new session variable of type `V`, mutating this session in place and returning it
    /// for chaining.
    ///
    /// Like [`register`](Self::register), the change is applied to the shared backing store, so it
    /// is observed through every clone of this session.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with_some<V: VortexSessionVar>(self, var: V) -> Self {
        if self.get_opt::<V>().is_some() {
            vortex_panic!(
                "Session variable of type {} already exists",
                type_name::<V>()
            );
        }
        self.register(var);
        self
    }

    /// Returns whether unknown plugins should deserialize as foreign placeholders.
    pub fn allows_unknown(&self) -> bool {
        self.get_opt::<UnknownPluginPolicy>()
            .is_some_and(|p| p.allow_unknown)
    }

    /// Allow deserializing unknown plugin IDs as non-executable foreign placeholders.
    ///
    /// Mutates this session in place and returns it for chaining.
    pub fn allow_unknown(self) -> Self {
        let mut policy = *self.get::<UnknownPluginPolicy>();
        policy.allow_unknown = true;
        self.register(policy);
        self
    }
}

impl SessionExt for VortexSession {
    fn session(&self) -> VortexSession {
        self.clone()
    }

    fn get<V: VortexSessionVar + Default>(&self) -> SessionGuard<'_, V> {
        if self.get_opt::<V>().is_none() {
            self.insert_default::<V>();
        }
        self.get_opt::<V>()
            .vortex_expect("variable was just inserted")
    }

    fn get_opt<V: VortexSessionVar>(&self) -> Option<SessionGuard<'_, V>> {
        let snapshot = self.0.load();
        snapshot
            .contains_key(&TypeId::of::<V>())
            .then(|| SessionGuard {
                snapshot,
                _session: PhantomData,
                _marker: PhantomData,
            })
    }

    fn get_mut<V: VortexSessionVar + Default + Clone>(&self) -> SessionMut<'_, V> {
        let value = (*self.get::<V>()).clone();
        SessionMut {
            session: self,
            value: Some(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::VortexSession;
    use crate::SessionExt;
    use crate::SessionVar;

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Counter {
        count: u32,
    }

    impl SessionVar for Counter {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[derive(Clone, Debug, Default)]
    struct Other;

    impl SessionVar for Other {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    thread_local! {
        /// Lets `Reentrant::default` reach back into the session it is being inserted into.
        static REENTRANT_SESSION: std::cell::RefCell<Option<VortexSession>> =
            const { std::cell::RefCell::new(None) };
    }

    #[derive(Clone, Debug)]
    struct Reentrant {
        inner: u32,
    }

    impl Default for Reentrant {
        fn default() -> Self {
            // Re-enter the *same* session while this default is being constructed, itself
            // triggering another default insertion. This is only sound because the default is
            // built outside the `rcu` closure (no lock held); if it ran under the closure this
            // would deadlock or recurse forever.
            REENTRANT_SESSION.with(|s| {
                if let Some(session) = s.borrow().as_ref() {
                    drop(session.get::<Counter>());
                }
            });
            Reentrant { inner: 7 }
        }
    }

    impl SessionVar for Reentrant {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn with_some_round_trip() {
        let session = VortexSession::empty().with_some(Counter { count: 1 });

        assert_eq!(*session.get::<Counter>(), Counter { count: 1 });
        assert!(session.get_opt::<Other>().is_none());
    }

    #[test]
    fn get_inserts_default() {
        let session = VortexSession::empty();
        assert!(session.get_opt::<Counter>().is_none());

        assert_eq!(session.get::<Counter>().count, 0);
        // The default was published, so it is now observable.
        assert!(session.get_opt::<Counter>().is_some());
    }

    #[test]
    fn register_is_visible_through_clones() {
        let session = VortexSession::empty();
        let clone = session.clone();

        session.register(Counter { count: 7 });

        // Registration mutates the shared backing store.
        assert_eq!(clone.get::<Counter>().count, 7);
    }

    #[test]
    fn with_some_mutates_shared_store() {
        let session = VortexSession::empty();
        let clone = session.clone();

        let configured = session.with_some(Counter { count: 5 });
        assert_eq!(configured.get::<Counter>().count, 5);

        // `with_some` mutates the shared backing store in place, so the clone observes it too.
        assert_eq!(clone.get::<Counter>().count, 5);
    }

    #[test]
    fn allow_unknown_mutates_shared_store() {
        let session = VortexSession::empty();
        let clone = session.clone();
        assert!(!clone.allows_unknown());

        session.allow_unknown();

        // The flag is flipped on the shared backing store.
        assert!(clone.allows_unknown());
    }

    #[test]
    fn empty_sessions_are_independent() {
        // Each `empty()` creates its own backing cell, so separately built sessions do not share
        // state.
        let session = VortexSession::empty().with_some(Counter { count: 1 });
        let other = VortexSession::empty().with_some(Counter { count: 2 });

        session.register(Counter { count: 9 });
        assert_eq!(session.get::<Counter>().count, 9);
        assert_eq!(other.get::<Counter>().count, 2);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn with_some_duplicate_panics() {
        VortexSession::empty()
            .with::<Counter>()
            .with_some(Counter { count: 1 });
    }

    #[test]
    fn allow_unknown_flag_is_opt_in() {
        let session = VortexSession::empty();
        assert!(!session.allows_unknown());

        let session = session.allow_unknown();
        assert!(session.allows_unknown());
    }

    #[test]
    fn get_opt_does_not_insert_a_default() {
        let session = VortexSession::empty();

        // Unlike `get`, `get_opt` is a pure read and never publishes a default.
        assert!(session.get_opt::<Counter>().is_none());
        assert!(session.get_opt::<Counter>().is_none());
    }

    #[test]
    fn default_insertion_may_reenter_the_session_without_deadlocking() {
        let session = VortexSession::empty();
        REENTRANT_SESSION.with(|s| *s.borrow_mut() = Some(session.clone()));

        // `get::<Reentrant>` inserts a default; building that default re-enters the same session
        // via `get::<Counter>` (another default insertion). Because each default is constructed
        // outside the `rcu` closure, both inserts complete instead of deadlocking.
        assert_eq!(session.get::<Reentrant>().inner, 7);
        assert!(session.get_opt::<Reentrant>().is_some());
        assert!(session.get_opt::<Counter>().is_some());

        REENTRANT_SESSION.with(|s| *s.borrow_mut() = None);
    }

    #[test]
    fn get_mut_publishes_on_drop() {
        let session = VortexSession::empty();
        session.register(Counter { count: 1 });

        session.get_mut::<Counter>().count = 42;

        assert_eq!(session.get::<Counter>().count, 42);
    }

    #[test]
    fn get_mut_inserts_default_then_mutates() {
        let session = VortexSession::empty();
        assert!(session.get_opt::<Counter>().is_none());

        session.get_mut::<Counter>().count += 5;

        assert_eq!(session.get::<Counter>().count, 5);
    }

    #[test]
    fn get_mut_mutation_is_visible_through_clones() {
        let session = VortexSession::empty().with_some(Counter { count: 1 });
        let clone = session.clone();

        session.get_mut::<Counter>().count = 9;

        // The mutated value is published copy-on-write to the shared backing store.
        assert_eq!(clone.get::<Counter>().count, 9);
    }
}
