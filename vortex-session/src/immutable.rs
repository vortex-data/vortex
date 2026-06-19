// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The immutable [`VortexSession`] and its [`VortexSessionBuilder`].
//!
//! [`VortexSession`] is backed by an immutable [`HashMap`]: once built, its set of session
//! variables can never change, so reads are lock-free and return plain references. Build all
//! mutable state in a [`VortexSessionBuilder`], then call [`VortexSessionBuilder::build`] to
//! freeze it into a [`VortexSession`].

use std::any::TypeId;
use std::any::type_name;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::IdHasher;
use crate::SessionVar;
use crate::UnknownPluginPolicy;

/// A [`SessionVar`] that can be stored in a [`VortexSession`].
pub trait VortexSessionVar: SessionVar {}

impl<V: SessionVar + Clone> VortexSessionVar for V {}

/// The type map backing a [`VortexSession`]. Immutable once wrapped in an `Arc`.
type VortexSessionVars = HashMap<TypeId, Box<dyn VortexSessionVar>, BuildHasherDefault<IdHasher>>;

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions,
/// dtypes, etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
///
/// The set of session variables is fixed at construction time, so lookups take no locks and
/// hand out plain references.
#[derive(Clone, Debug)]
pub struct VortexSession(Arc<VortexSessionVars>);

impl VortexSession {
    /// Create a new [`VortexSession`] with no session state.
    pub fn empty() -> Self {
        Self(Default::default())
    }

    /// Create an empty [`VortexSessionBuilder`].
    pub fn builder() -> VortexSessionBuilder {
        VortexSessionBuilder::default()
    }

    /// Returns the session variable of type `V`.
    ///
    /// # Panics
    ///
    /// If no variable of that type exists in this session.
    #[expect(
        clippy::same_name_method,
        reason = "SessionExt re-exposes get/get_opt so generic `S: SessionExt` code can call them"
    )]
    pub fn get<V: VortexSessionVar>(&self) -> &V {
        self.get_opt::<V>().unwrap_or_else(|| {
            vortex_panic!("Session variable of type {} not found", type_name::<V>())
        })
    }

    /// Returns the session variable of type `V`, if it exists.
    #[expect(
        clippy::same_name_method,
        reason = "SessionExt re-exposes get/get_opt so generic `S: SessionExt` code can call them"
    )]
    pub fn get_opt<V: VortexSessionVar>(&self) -> Option<&V> {
        self.0.get(&TypeId::of::<V>()).map(|var| {
            (**var)
                .as_any()
                .downcast_ref::<V>()
                .vortex_expect("Type mismatch - this is a bug")
        })
    }

    /// Returns whether unknown plugins should deserialize as foreign placeholders.
    pub fn allows_unknown(&self) -> bool {
        self.get_opt::<UnknownPluginPolicy>()
            .is_some_and(|p| p.allow_unknown)
    }
}

/// A mutable builder for [`VortexSession`].
///
/// Holds a private, mutable copy of a session's state. Modify it freely, then call
/// [`VortexSessionBuilder::build`] to freeze it into a new immutable [`VortexSession`].
#[derive(Debug, Default)]
pub struct SessionBuilder {
    vars: VortexSessionVars,
}

/// Public builder type for constructing [`VortexSession`] values.
pub type VortexSessionBuilder = SessionBuilder;

impl SessionBuilder {
    /// Inserts a new session variable of type `V` with its default value.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with<V: VortexSessionVar + Default>(self) -> Self {
        self.with_some(V::default())
    }

    /// Inserts a new session variable of type `V`.
    ///
    /// # Panics
    ///
    /// If a variable of that type already exists.
    pub fn with_some<V: VortexSessionVar>(mut self, var: V) -> Self {
        match self.vars.entry(TypeId::of::<V>()) {
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

    /// Returns a mutable reference to the session variable of type `V`, inserting a default
    /// one if it does not exist.
    pub fn get_mut<V: VortexSessionVar + Default>(&mut self) -> &mut V {
        self.vars
            .entry(TypeId::of::<V>())
            .or_insert_with(|| Box::new(V::default()))
            .as_any_mut()
            .downcast_mut::<V>()
            .vortex_expect("Type mismatch - this is a bug")
    }

    /// Returns a mutable reference to the session variable of type `V`, if it exists.
    pub fn get_mut_opt<V: VortexSessionVar>(&mut self) -> Option<&mut V> {
        self.vars.get_mut(&TypeId::of::<V>()).map(|var| {
            var.as_any_mut()
                .downcast_mut::<V>()
                .vortex_expect("Type mismatch - this is a bug")
        })
    }

    /// Allow deserializing unknown plugin IDs as non-executable foreign placeholders.
    pub fn allow_unknown(mut self) -> Self {
        self.get_mut::<UnknownPluginPolicy>().allow_unknown = true;
        self
    }

    /// Finalize this builder into an immutable [`VortexSession`].
    pub fn build(self) -> VortexSession {
        VortexSession(Arc::new(self.vars))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::VortexSession;
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

    #[test]
    fn builder_round_trip() {
        let session = VortexSession::builder()
            .with_some(Counter { count: 1 })
            .build();

        assert_eq!(session.get::<Counter>(), &Counter { count: 1 });
        assert!(session.get_opt::<Other>().is_none());
    }

    #[test]
    fn get_mut_inserts_default() {
        let mut builder = VortexSession::builder();
        assert!(builder.get_mut_opt::<Counter>().is_none());

        builder.get_mut::<Counter>().count = 42;
        let session = builder.build();

        assert_eq!(session.get::<Counter>().count, 42);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn with_some_duplicate_panics() {
        VortexSession::builder()
            .with::<Counter>()
            .with_some(Counter { count: 1 });
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        VortexSession::empty().get::<Counter>();
    }

    #[test]
    fn allow_unknown_flag_is_opt_in() {
        let session = VortexSession::empty();
        assert!(!session.allows_unknown());

        let session = VortexSession::builder().allow_unknown().build();
        assert!(session.allows_unknown());
    }
}
