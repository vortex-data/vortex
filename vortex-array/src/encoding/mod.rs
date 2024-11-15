//! Traits and types to define shared unique encoding identifiers.

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use vortex_error::{vortex_panic, VortexResult};

use crate::canonical::{Canonical, IntoCanonical};
use crate::stats::ArrayStatistics as _;
use crate::{ArrayData, ArrayDef, ArrayTrait};

pub mod opaque;

// TODO(robert): Outline how you create a well known encoding id

/// EncodingId is a unique name and numerical code of the array
///
/// 0x0000 - reserved marker encoding
/// 0x0001 - 0x0400 - vortex internal encodings (1 - 1024)
/// 0x0401 - 0x7FFF - well known extension encodings (1025 - 32767)
/// 0x8000 - 0xFFFF - custom extension encodings (32768 - 65535)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EncodingId(&'static str, u16);

impl EncodingId {
    pub const fn new(id: &'static str, code: u16) -> Self {
        Self(id, code)
    }

    pub const fn code(&self) -> u16 {
        self.1
    }
}

impl Display for EncodingId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:#04x})", self.0, self.1)
    }
}

impl AsRef<str> for EncodingId {
    fn as_ref(&self) -> &str {
        self.0
    }
}

pub type EncodingRef = &'static dyn ArrayEncoding;

/// Object-safe encoding trait for an array.
pub trait ArrayEncoding: 'static + Sync + Send + Debug {
    fn id(&self) -> EncodingId;

    /// Flatten the given array.
    fn canonicalize(&self, array: ArrayData) -> VortexResult<Canonical>;

    /// Unwrap the provided array into an implementation of ArrayTrait
    fn with_dyn(
        &self,
        array: &ArrayData,
        f: &mut dyn for<'b> FnMut(&'b (dyn ArrayTrait + 'b)) -> VortexResult<()>,
    ) -> VortexResult<()>;
}

impl PartialEq for dyn ArrayEncoding + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for dyn ArrayEncoding + '_ {}
impl Hash for dyn ArrayEncoding + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

/// Non-object-safe extensions to the ArrayEncoding trait.
pub trait ArrayEncodingExt {
    type D: ArrayDef;

    fn into_canonical(array: ArrayData) -> VortexResult<Canonical> {
        // get the stats before converting to the typed array
        let stats = array.statistics().to_set();

        let typed = <<Self::D as ArrayDef>::Array as TryFrom<ArrayData>>::try_from(array)?;
        let canonical = IntoCanonical::into_canonical(typed)?;

        // propagate the stats to the canonicalized array
        for (stat, scalar) in stats {
            canonical.statistics().set(stat, scalar);
        }
        Ok(canonical)
    }

    fn with_dyn<R, F>(array: &ArrayData, mut f: F) -> R
    where
        F: for<'b> FnMut(&'b (dyn ArrayTrait + 'b)) -> R,
    {
        let typed = <<Self::D as ArrayDef>::Array as TryFrom<ArrayData>>::try_from(array.clone())
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to convert array to {}",
                    std::any::type_name::<<Self::D as ArrayDef>::Array>()
                )
            });
        f(&typed)
    }
}

pub trait ArrayEncodingRef {
    fn encoding(&self) -> EncodingRef;
}

#[doc = "Encoding ID constants for all Vortex-provided encodings"]
#[allow(dead_code)]
pub mod ids {
    // reserved - 0x0000
    pub(crate) const RESERVED: u16 = 0;

    // Vortex built-in encodings (1 - 15)
    // built-ins first
    pub const NULL: u16 = 1;
    pub const BOOL: u16 = 2;
    pub const PRIMITIVE: u16 = 3;
    pub const STRUCT: u16 = 4;
    pub const VAR_BIN: u16 = 5;
    pub const VAR_BIN_VIEW: u16 = 6;
    pub const EXTENSION: u16 = 7;
    pub const SPARSE: u16 = 8;
    pub const CONSTANT: u16 = 9;
    pub const CHUNKED: u16 = 10;

    // currently unused, saved for future built-ins
    // e.g., List, FixedList, Union, Tensor, etc.
    pub(crate) const RESERVED_11: u16 = 11;
    pub(crate) const RESERVED_12: u16 = 12;
    pub(crate) const RESERVED_13: u16 = 13;
    pub(crate) const RESERVED_14: u16 = 14;
    pub(crate) const RESERVED_15: u16 = 15;
    pub(crate) const RESERVED_16: u16 = 16;

    // bundled extensions
    pub const ALP: u16 = 17;
    pub const BYTE_BOOL: u16 = 18;
    pub const DATE_TIME_PARTS: u16 = 19;
    pub const DICT: u16 = 20;
    pub const FL_BITPACKED: u16 = 21;
    pub const FL_DELTA: u16 = 22;
    pub const FL_FOR: u16 = 23;
    pub const FSST: u16 = 24;
    pub const ROARING_BOOL: u16 = 25;
    pub const ROARING_INT: u16 = 26;
    pub const RUN_END: u16 = 27;
    pub const RUN_END_BOOL: u16 = 28;
    pub const ZIGZAG: u16 = 29;
    pub const ALP_RD: u16 = 30;
}

#[cfg(test)]
mod tests {
    use super::ids;
    use crate::aliases::hash_set::HashSet;

    #[test]
    fn test_encoding_id() {
        let all_ids = [
            ids::RESERVED,
            ids::NULL,
            ids::BOOL,
            ids::PRIMITIVE,
            ids::STRUCT,
            ids::VAR_BIN,
            ids::VAR_BIN_VIEW,
            ids::EXTENSION,
            ids::SPARSE,
            ids::CONSTANT,
            ids::CHUNKED,
            ids::RESERVED_11,
            ids::RESERVED_12,
            ids::RESERVED_13,
            ids::RESERVED_14,
            ids::RESERVED_15,
            ids::RESERVED_16,
            ids::ALP,
            ids::BYTE_BOOL,
            ids::DATE_TIME_PARTS,
            ids::DICT,
            ids::FL_BITPACKED,
            ids::FL_DELTA,
            ids::FL_FOR,
            ids::FSST,
            ids::ROARING_BOOL,
            ids::ROARING_INT,
            ids::RUN_END,
            ids::RUN_END_BOOL,
            ids::ZIGZAG,
        ];

        let mut ids_set = HashSet::with_capacity(all_ids.len());
        ids_set.extend(all_ids);
        assert_eq!(ids_set.len(), all_ids.len()); // no duplicates
        assert!(ids_set.iter().max().unwrap() <= &0x0400); // no ids are greater than 1024
        for (i, id) in all_ids.iter().enumerate() {
            // monotonic with no gaps
            assert_eq!(i as u16, *id, "id at index {} is not equal to index", i);
        }
    }
}
