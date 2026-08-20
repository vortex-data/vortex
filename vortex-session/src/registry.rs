// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use lasso::Spur;
use lasso::ThreadedRodeo;
use parking_lot::RwLock;
use vortex_error::VortexExpect;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

/// Global string interner for [`Id`] values.
static INTERNER: LazyLock<ThreadedRodeo<Spur, DefaultHashBuilder>> =
    LazyLock::new(|| ThreadedRodeo::with_hasher(DefaultHashBuilder::default()));

/// A lightweight, copyable identifier backed by a global string interner.
///
/// Used for array encoding IDs, scalar function IDs, layout IDs, and similar
/// globally-unique string identifiers throughout Vortex. Equality and hashing
/// are O(1) symbol comparisons.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(Spur);

impl Id {
    /// Intern a string and return its `Id`.
    pub fn new(s: &str) -> Self {
        Self(INTERNER.get_or_intern(s))
    }

    /// Intern a string and return its `Id`.
    pub fn new_static(s: &'static str) -> Self {
        Self(INTERNER.get_or_intern_static(s))
    }

    /// Returns the interned string.
    pub fn as_str(&self) -> &str {
        let s = INTERNER.resolve(&self.0);
        // SAFETY: INTERNER is 'static and its arena is append-only, so resolved string
        // pointers are stable for the lifetime of the program.
        unsafe { &*(s as *const str) }
    }
}

impl From<&str> for Id {
    #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Debug for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Id(\"{}\")", self.as_str())
    }
}

impl PartialOrd for Id {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Id {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl AsRef<str> for Id {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<&Id> for Id {
    fn eq(&self, other: &&Id) -> bool {
        self == *other
    }
}

impl PartialEq<Id> for &Id {
    fn eq(&self, other: &Id) -> bool {
        *self == other
    }
}

/// A lazily-initialized, cached [`Id`] for use as a `static`.
///
/// Avoids repeated interner write-lock acquisition by storing the interned [`Id`]
/// on first access and returning the cached copy on all subsequent calls.
///
/// # Example
///
/// ```
/// use vortex_session::registry::{CachedId, Id};
///
/// static MY_ID: CachedId = CachedId::new("my.encoding");
///
/// fn get_id() -> Id {
///     *MY_ID
/// }
/// ```
pub struct CachedId {
    s: &'static str,
    cached: OnceLock<Id>,
}

impl CachedId {
    /// Create a new `CachedId` that will intern `s` on first access.
    pub const fn new(s: &'static str) -> Self {
        Self {
            s,
            cached: OnceLock::new(),
        }
    }
}

impl Deref for CachedId {
    type Target = Id;

    #[expect(
        clippy::disallowed_methods,
        reason = "CachedId interns its static id once here"
    )]
    fn deref(&self) -> &Id {
        self.cached.get_or_init(|| Id::new_static(self.s))
    }
}

/// A [`ReadContext`] holds a set of interned IDs for use during deserialization, mapping
/// u16 indices to IDs.
#[derive(Clone, Debug)]
pub struct ReadContext {
    ids: Arc<[Id]>,
}

impl ReadContext {
    /// Create a context with the given initial IDs.
    pub fn new(ids: impl Into<Arc<[Id]>>) -> Self {
        Self { ids: ids.into() }
    }

    /// Resolve an interned ID by its index.
    pub fn resolve(&self, idx: u16) -> Option<Id> {
        self.ids.get(idx as usize).cloned()
    }

    pub fn ids(&self) -> &[Id] {
        &self.ids
    }
}

/// An [`Interner`] holds a set of interned IDs for use during serialization/deserialization,
/// mapping IDs to u16 indices.
///
/// ## Upcoming Changes
///
/// This object holds an Arc of RwLock internally because we need concurrent access from the
/// layout writer code path. We should update SegmentSink to take an Array rather than
/// ByteBuffer such that serializing arrays is done sequentially.
#[derive(Clone, Debug, Default)]
pub struct Interner {
    // TODO(ngates): it's a long story, but if we make SegmentSink and SegmentSource take an
    //  enum of Segment { Array, DType, Buffer } then we don't actually need a mutable context
    //  in the LayoutWriter, therefore we don't need a RwLock here and everyone is happier.
    ids: Arc<RwLock<Vec<Id>>>,
    // Optional set of permissible IDs; when present, only these may be interned.
    allowed: Option<Arc<HashSet<Id>>>,
}

impl Interner {
    /// Create an interner with the given initial IDs.
    pub fn new(ids: Vec<Id>) -> Self {
        Self {
            ids: Arc::new(RwLock::new(ids)),
            allowed: None,
        }
    }

    /// Create an empty interner.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Restrict the permissible set of interned IDs to `allowed`.
    ///
    /// The set is snapshotted at this call: IDs registered elsewhere afterwards are not
    /// permitted.
    pub fn with_allowed_ids(mut self, allowed: HashSet<Id>) -> Self {
        self.allowed = Some(Arc::new(allowed));
        self
    }

    /// Intern an ID, returning its index.
    pub fn intern(&self, id: &Id) -> Option<u16> {
        if let Some(allowed) = &self.allowed
            && !allowed.contains(id)
        {
            // ID not permitted, cannot intern.
            return None;
        }

        let mut ids = self.ids.write();
        if let Some(idx) = ids.iter().position(|e| e == id) {
            return Some(u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX items"));
        }

        let idx = ids.len();
        assert!(
            idx < u16::MAX as usize,
            "Cannot have more than u16::MAX items"
        );
        ids.push(*id);
        Some(u16::try_from(idx).vortex_expect("checked already"))
    }

    /// Get the list of interned IDs.
    pub fn to_ids(&self) -> Vec<Id> {
        self.ids.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use vortex_utils::aliases::hash_set::HashSet;

    use super::CachedId;
    use super::Interner;

    static VALID: CachedId = CachedId::new("vortex.test.valid");
    static INVALID: CachedId = CachedId::new("vortex.test.invalid");

    #[test]
    fn context_filters_interned_ids() {
        let valid = *VALID;
        let invalid = *INVALID;
        let context = Interner::empty().with_allowed_ids(HashSet::from([valid]));

        assert_eq!(context.intern(&valid), Some(0));
        assert_eq!(context.intern(&valid), Some(0));
        assert_eq!(context.intern(&invalid), None);
        assert_eq!(context.to_ids(), [valid]);
    }
}
