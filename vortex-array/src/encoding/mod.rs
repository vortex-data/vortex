//! Traits and types to define shared unique encoding identifiers.

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::vtable::{
    CanonicalVTable, ComputeVTable, EncodingVTable, MetadataVTable, StatisticsVTable,
    ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use crate::{DeserializeMetadata, SerializeMetadata};

pub mod opaque;

// TODO(robert): Outline how you create a well known encoding id

/// EncodingId is a unique name and numerical code of the array
///
/// 0x0000 - reserved marker encoding
/// 0x0001 - 0x0400 - vortex internal encodings (1 - 1024)
/// 0x0401 - 0x7FFF - well known extension encodings (1025 - 32767)
/// 0x8000 - 0xFFFF - custom extension encodings (32768 - 65535)
#[derive(Clone, Copy, Debug, Eq)]
pub struct EncodingId(&'static str, u16);

impl EncodingId {
    pub const fn new(id: &'static str, code: u16) -> Self {
        Self(id, code)
    }

    pub const fn code(&self) -> u16 {
        self.1
    }
}

// The encoding is identified only by its numeric ID, so we only use that for PartialEq and Hash
impl PartialEq for EncodingId {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Hash for EncodingId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.1.hash(state);
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

/// Marker trait for array encodings with their associated Array type.
pub trait Encoding: 'static {
    const ID: EncodingId;

    type Array;
    type Metadata: SerializeMetadata + DeserializeMetadata;
}

pub type EncodingRef = &'static dyn EncodingVTable;

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
    pub const CONSTANT: u16 = 9;
    pub const CHUNKED: u16 = 10;
    pub const LIST: u16 = 11;

    // currently unused, saved for future built-ins
    // e.g., FixedList, Union, Tensor, etc.
    pub(crate) const RESERVED_12: u16 = 12;
    pub(crate) const RESERVED_13: u16 = 13;
    pub(crate) const RESERVED_14: u16 = 14;
    pub(crate) const RESERVED_15: u16 = 15;
    pub(crate) const RESERVED_16: u16 = 16;

    // bundled extensions
    pub const ALP: u16 = 17;
    pub const ALP_RD: u16 = 18;
    pub const BYTE_BOOL: u16 = 19;
    pub const DATE_TIME_PARTS: u16 = 20;
    pub const DICT: u16 = 21;
    pub const FL_BITPACKED: u16 = 22;
    pub const FL_DELTA: u16 = 23;
    pub const FL_FOR: u16 = 24;
    pub const FL_RLE: u16 = 25;
    pub const FSST: u16 = 26;
    pub const RUN_END: u16 = 27;
    pub const SPARSE: u16 = 8;
    pub const ZIGZAG: u16 = 28;
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::{ids, EncodingId};
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
            ids::LIST,
            ids::RESERVED_12,
            ids::RESERVED_13,
            ids::RESERVED_14,
            ids::RESERVED_15,
            ids::RESERVED_16,
            ids::ALP,
            ids::ALP_RD,
            ids::BYTE_BOOL,
            ids::DATE_TIME_PARTS,
            ids::DICT,
            ids::FL_BITPACKED,
            ids::FL_DELTA,
            ids::FL_FOR,
            ids::FL_RLE,
            ids::FSST,
            ids::RUN_END,
            ids::ZIGZAG,
        ];

        // make sure we didn't forget any ids
        assert_eq!(all_ids.len(), ids::ZIGZAG as usize + 1);

        let mut ids_set = HashSet::with_capacity(all_ids.len());
        ids_set.extend(all_ids);
        assert_eq!(ids_set.len(), all_ids.len()); // no duplicates
        assert!(ids_set.iter().max().unwrap() <= &0x0400); // no ids are greater than 1024
        for (i, id) in all_ids.iter().enumerate() {
            // monotonic with no gaps
            assert_eq!(i as u16, *id, "id at index {} is not equal to index", i);
        }
    }

    #[test]
    fn test_encoding_id_eq() {
        let fizz = EncodingId::new("fizz", 0);
        let buzz = EncodingId::new("buzz", 0);
        let fizzbuzz = EncodingId::new("fizzbuzz", 1);

        assert_eq!(fizz, buzz);
        assert_ne!(fizz, fizzbuzz);
    }
}
