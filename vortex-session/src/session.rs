// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`VortexSession`] container.
//!
//! A [`VortexSession`] is a type-map of [`VortexSessionVar`]s keyed by [`TypeId`]. It is backed by
//! an [`ArcSwapMap`], giving lock-free reads and copy-on-write writes:
//!
//! * **Reads** ([`SessionExt::get`], [`SessionExt::get_opt`]) load the current snapshot
//!   without taking any lock and hand back a [`SessionGuard`] that derefs to the variable. Because a
//!   read never takes a lock, it can never deadlock, and because it never holds a lock across the
//!   returned reference there is no reader/writer contention.
//!
//! * **Writes** ([`VortexSession::with_some`], and [`SessionExt::get`] or
//!   [`SessionExt::get_or_insert_with`] on a missing variable) are copy-on-write: the map is cloned,
//!   the change applied to the private copy, and the new map atomically published. The value is
//!   constructed *before* the map is updated, so no user code (in particular, no `Default::default`
//!   implementation or initializer closure) ever runs while a lock is held. This is the key
//!   difference from the previous `DashMap`-backed session, where `entry().or_insert_with(f)` ran
//!   `f` while holding the shard's write lock and could deadlock if `f` re-entered the session.
//!
//! A modified session is produced by mutating it **in place**: [`VortexSession::with_some`] — and
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
use std::sync::Arc;

use arc_swap::Guard;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArcSwapMap;
use crate::IdHasher;
use crate::SessionExt;
use crate::SessionVar;
use crate::UnknownPluginPolicy;

/// A [`SessionVar`] that can be stored in a [`VortexSession`].
///
/// This trait is implemented automatically for every [`SessionVar`]. Snapshots store shared
/// references to variables, so their types do not need to implement [`Clone`]. A variable that
/// exposes shared mutable state must synchronize that state in its own API.
pub trait VortexSessionVar: SessionVar {}

impl<V: SessionVar> VortexSessionVar for V {}

/// The hasher for the session type-map; [`TypeId`]s are already hashes.
type IdHashBuilder = BuildHasherDefault<IdHasher>;

/// The immutable type-map backing a published [`VortexSession`] snapshot.
type SessionVars = HashMap<TypeId, Arc<dyn VortexSessionVar>, IdHashBuilder>;

/// The shared, copy-on-write store that publishes [`SessionVars`] snapshots.
type SharedSessionVars = ArcSwapMap<TypeId, Arc<dyn VortexSessionVar>, IdHashBuilder>;

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

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions,
/// dtypes, etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
///
/// Cloning a session is cheap and shares the backing store: a variable registered through one
/// clone (via [`VortexSession::with_some`] or one of the `with_*` helpers) is observed by all
/// clones. To build an *independent* session, start from [`VortexSession::empty`].
#[derive(Clone)]
pub struct VortexSession(SharedSessionVars);

impl Debug for VortexSession {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0
            .read(|vars| f.debug_tuple("VortexSession").field(vars).finish())
    }
}

impl VortexSession {
    /// Create a new [`VortexSession`] with no session state.
    pub fn empty() -> Self {
        Self(SharedSessionVars::default())
    }

    /// Inserts a session variable of type `V`, unconditionally replacing any existing variable of
    /// that type.
    ///
    /// This replaces the whole value; it does not merge changes into the previous value. Existing
    /// [`SessionGuard`]s keep observing their pinned snapshot, while new reads see the replacement.
    /// Variables that expose shared mutable state must synchronize it in their own API.
    ///
    /// [`with_some`](Self::with_some) and [`get_or_insert_with`](SessionExt::get_or_insert_with)
    /// use [`ArcSwapMap::insert_if_absent`](crate::arc_swap_map::ArcSwapMap::insert_if_absent) so
    /// that a first registration cannot overwrite another.
    pub fn register<V: VortexSessionVar>(&self, var: V) {
        self.0.insert(TypeId::of::<V>(), Arc::new(var));
    }

    /// Inserts a new session variable of type `V` with its default value, mutating this session in
    /// place and returning it for chaining.
    ///
    /// The change is applied copy-on-write to the shared backing store, so it is observed through
    /// every clone of this session.
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
    /// The change is applied copy-on-write to the shared backing store, so it is observed through
    /// every clone of this session.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with_some<V: VortexSessionVar>(self, var: V) -> Self {
        if !self.0.insert_if_absent(TypeId::of::<V>(), Arc::new(var)) {
            vortex_panic!(
                "Session variable of type {} already exists",
                type_name::<V>()
            );
        }
        self
    }

    /// Returns whether unknown plugins should deserialize as foreign placeholders.
    pub fn allows_unknown(&self) -> bool {
        self.get_opt::<UnknownPluginPolicy>()
            .is_some_and(|p| p.allow_unknown)
    }

    /// Allow deserializing unknown plugin IDs as non-executable foreign placeholders.
    ///
    /// Sets the policy to allow unknown plugin IDs.
    pub fn allow_unknown(&self) {
        self.register(UnknownPluginPolicy {
            allow_unknown: true,
        });
    }
}

impl SessionExt for VortexSession {
    fn session(&self) -> VortexSession {
        self.clone()
    }

    fn get<V: VortexSessionVar + Default>(&self) -> SessionGuard<'_, V> {
        self.get_or_insert_with(V::default)
    }

    fn get_or_insert_with<V: VortexSessionVar>(
        &self,
        init: impl FnOnce() -> V,
    ) -> SessionGuard<'_, V> {
        if let Some(var) = self.get_opt::<V>() {
            return var;
        }

        // NB: `init` runs before the map is updated, never inside it, because it may re-enter the
        // session. If a concurrent writer published a value first, that value is kept.
        let var: Arc<dyn VortexSessionVar> = Arc::new(init());
        self.0.insert_if_absent(TypeId::of::<V>(), var);

        self.get_opt::<V>()
            .vortex_expect("a variable of type `V` is present after `insert_if_absent`")
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
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;
    use std::sync::Barrier;
    use std::thread;

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

    #[derive(Debug, Default)]
    struct NonClone {
        count: u32,
    }

    impl SessionVar for NonClone {
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
    fn test_non_clone_session_var_can_be_registered_and_read() {
        let session = VortexSession::empty().with_some(NonClone { count: 3 });

        assert_eq!(session.get::<NonClone>().count, 3);
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
    fn get_or_insert_with_inserts_only_when_absent() {
        let session = VortexSession::empty();

        assert_eq!(session.get_or_insert_with(|| Counter { count: 3 }).count, 3);
        assert_eq!(
            session
                .get_or_insert_with::<Counter>(|| panic!("the variable is already present"))
                .count,
            3
        );
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
    fn get_opt_does_not_insert_a_default() {
        let session = VortexSession::empty();

        // Unlike `get`, `get_opt` is a pure read and never publishes a default.
        assert!(session.get_opt::<Counter>().is_none());
        assert!(session.get_opt::<Counter>().is_none());
    }

    #[test]
    fn inserting_a_default_while_holding_a_guard_succeeds() {
        let session = VortexSession::empty().with_some(Counter { count: 5 });

        // Hold a read guard for one variable...
        let counter = session.get::<Counter>();

        // ...then read a *missing* variable, which inserts its default copy-on-write via `rcu`
        // while `counter` is still alive. A `SessionGuard` is a lock-free snapshot, not a lock, so
        // the write proceeds without waiting on the outstanding guard rather than deadlocking.
        let other = session.get::<Other>();

        // The held guard still observes the snapshot it was read from (which arc-swap keeps alive
        // for the guard's lifetime), and the freshly read guard sees the just-inserted default.
        assert_eq!(counter.count, 5);
        let _: &Other = &other;

        // The inserted default is also observable through a subsequent independent read.
        assert!(session.get_opt::<Other>().is_some());
    }

    #[test]
    fn a_held_guard_keeps_observing_its_own_snapshot_after_a_write() {
        let session = VortexSession::empty().with_some(Counter { count: 1 });

        let held = session.get::<Counter>();
        session.register(Counter { count: 2 });

        // The held guard pins the snapshot it was read from, so it still sees the old value, while
        // a fresh read sees the newly published one.
        assert_eq!(held.count, 1);
        assert_eq!(session.get::<Counter>().count, 2);
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
    fn concurrent_with_some_same_type_allows_only_one_winner() {
        for _ in 0..200 {
            let session = VortexSession::empty();
            let clone_a = session.clone();
            let clone_b = session.clone();
            let barrier = Barrier::new(2);

            let (res_a, res_b) = thread::scope(|s| {
                let h_a = s.spawn(|| {
                    barrier.wait();
                    catch_unwind(AssertUnwindSafe(|| clone_a.with_some(Counter { count: 1 })))
                });
                let h_b = s.spawn(|| {
                    barrier.wait();
                    catch_unwind(AssertUnwindSafe(|| clone_b.with_some(Counter { count: 2 })))
                });
                (h_a.join().unwrap(), h_b.join().unwrap())
            });

            let (winner, loser, expected) = match (res_a, res_b) {
                (Ok(winner), Err(panic)) => (winner, panic, 1),
                (Err(panic), Ok(winner)) => (winner, panic, 2),
                (Ok(_), Ok(_)) => {
                    panic!("both concurrent with_some calls succeeded for same type")
                }
                (Err(_), Err(_)) => {
                    panic!("both concurrent with_some calls panicked for same type")
                }
            };
            let msg = loser
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| loser.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(msg.contains("already exists"), "unexpected panic: {msg}");
            assert_eq!(winner.get::<Counter>().count, expected);
            assert_eq!(session.get::<Counter>().count, expected);
        }
    }
}
