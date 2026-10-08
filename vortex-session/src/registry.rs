// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::num::NonZeroU32;
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

/// Array encoding IDs reserved for the canonical encodings, in the order of their index in
/// [`Id::canonical_index`].
pub const CANONICAL_ARRAY_IDS: [&str; CANONICAL_LEN as usize] = [
    "vortex.null",
    "vortex.bool",
    "vortex.primitive",
    "vortex.decimal",
    "vortex.struct",
    "vortex.union",
    "vortex.listview",
    "vortex.map",
    "vortex.fixed_size_list",
    "vortex.varbinview",
    "vortex.variant",
    "vortex.ext",
];

/// Array encoding ID reserved for the constant encoding.
pub const CONSTANT_ARRAY_ID: &str = "vortex.constant";

const CANONICAL_LEN: u32 = 12;

/// Number of interner keys reserved for [`CANONICAL_ARRAY_IDS`] followed by [`CONSTANT_ARRAY_ID`].
const RESERVED_LEN: u32 = CANONICAL_LEN + 1;

/// Global string interner for [`Id`] values.
///
/// The reserved IDs are interned first, so they take the first keys in a fixed order and
/// [`Id::reserved`] can build them at compile time.
static INTERNER: LazyLock<ThreadedRodeo<Spur, DefaultHashBuilder>> = LazyLock::new(|| {
    let interner = ThreadedRodeo::with_hasher(DefaultHashBuilder::default());
    for name in CANONICAL_ARRAY_IDS.into_iter().chain([CONSTANT_ARRAY_ID]) {
        interner.get_or_intern_static(name);
    }
    interner
});

/// A lightweight, copyable identifier backed by a global string interner.
///
/// Used for array encoding IDs, scalar function IDs, layout IDs, and similar
/// globally-unique string identifiers throughout Vortex. Equality and hashing
/// are O(1) symbol comparisons.
///
/// The names in [`CANONICAL_ARRAY_IDS`] and [`CONSTANT_ARRAY_ID`] take the first interner keys,
/// so [`Id::reserved`] is a `const fn` and [`Id::is_canonical`], [`Id::is_constant`] and
/// [`Id::canonical_index`] are a single comparison of the key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(Spur);

impl Id {
    /// Returns the `Id` of a reserved name at compile time.
    ///
    /// # Panics
    ///
    /// Panics, at compile time when used in a `const`, if `name` is not one of
    /// [`CANONICAL_ARRAY_IDS`] or [`CONSTANT_ARRAY_ID`].
    pub const fn reserved(name: &str) -> Self {
        let mut index = 0;
        while index < CANONICAL_LEN {
            if str_eq(name, CANONICAL_ARRAY_IDS[index as usize]) {
                return Self::from_key(index + 1);
            }
            index += 1;
        }
        assert!(
            str_eq(name, CONSTANT_ARRAY_ID),
            "Id::reserved called with a name that is not reserved"
        );
        Self::from_key(RESERVED_LEN)
    }

    /// Builds the `Id` for the one-based interner key `key`, which must be at least 1.
    const fn from_key(key: u32) -> Self {
        let key = NonZeroU32::MIN.saturating_add(key - 1);
        // SAFETY: `Spur` is a `#[repr(transparent)]` wrapper around the one-based `NonZeroU32`
        // key, and the reserved names were interned first, in order, so they own keys
        // `1..=RESERVED_LEN`.
        Self(unsafe { std::mem::transmute::<NonZeroU32, Spur>(key) })
    }

    #[inline]
    fn key(&self) -> u32 {
        self.0.into_inner().get()
    }

    /// Whether this is one of the [`CANONICAL_ARRAY_IDS`].
    #[inline]
    pub fn is_canonical(&self) -> bool {
        self.key() < RESERVED_LEN
    }

    /// Whether this is the [`CONSTANT_ARRAY_ID`].
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.key() == RESERVED_LEN
    }

    /// Whether this is one of the [`CANONICAL_ARRAY_IDS`] or the [`CONSTANT_ARRAY_ID`].
    #[inline]
    pub fn is_canonical_or_constant(&self) -> bool {
        self.key() <= RESERVED_LEN
    }

    /// Returns the index of this `Id` in [`CANONICAL_ARRAY_IDS`], if it is canonical.
    #[inline]
    pub fn canonical_index(&self) -> Option<usize> {
        self.is_canonical().then(|| self.key() as usize - 1)
    }

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

const fn str_eq(lhs: &str, rhs: &str) -> bool {
    let (lhs, rhs) = (lhs.as_bytes(), rhs.as_bytes());
    if lhs.len() != rhs.len() {
        return false;
    }
    let mut i = 0;
    while i < lhs.len() {
        if lhs[i] != rhs[i] {
            return false;
        }
        i += 1;
    }
    true
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
    #[inline]
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
        if self
            .allowed
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(id))
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

    use super::CANONICAL_ARRAY_IDS;
    use super::CONSTANT_ARRAY_ID;
    use super::CachedId;
    use super::Id;
    use super::Interner;

    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "comparing interned and reserved ids"
    )]
    fn reserved_ids_match_interned_ids() {
        for (index, name) in CANONICAL_ARRAY_IDS.into_iter().enumerate() {
            let id = Id::reserved(name);
            assert_eq!(id, Id::new(name));
            assert_eq!(id.as_str(), name);
            assert!(id.is_canonical() && id.is_canonical_or_constant() && !id.is_constant());
            assert_eq!(id.canonical_index(), Some(index));
        }

        let constant = Id::reserved(CONSTANT_ARRAY_ID);
        assert_eq!(constant, Id::new(CONSTANT_ARRAY_ID));
        assert_eq!(constant.as_str(), CONSTANT_ARRAY_ID);
        assert!(constant.is_constant() && constant.is_canonical_or_constant());
        assert!(!constant.is_canonical());
        assert_eq!(constant.canonical_index(), None);

        let other = Id::new("vortex.test.unreserved");
        assert!(!other.is_canonical_or_constant());
        assert_eq!(other.canonical_index(), None);
    }

    #[test]
    #[should_panic(expected = "not reserved")]
    fn unreserved_name_panics() {
        let _id = Id::reserved("vortex.test.unreserved");
    }

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
