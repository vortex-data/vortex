// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt;

use vortex_dtype::DType;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Scalar;
use crate::ScalarValue;
use crate::extension::ExtScalarVTable;
use crate::extension::ExtScalarValue;

// --- Metadata ---

/// Metadata storing the expected number of characters (via `chars().count()`) for validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphemeMetadata {
    grapheme_count: usize,
}

impl fmt::Display for GraphemeMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "graphemes={}", self.grapheme_count)
    }
}

// --- Value type ---

/// A borrowed view of a grapheme-counted UTF-8 value, carrying both the string and its character
/// count from the extension metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphemeStr<'a> {
    text: &'a str,
    grapheme_count: usize,
}

impl PartialOrd for GraphemeStr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphemeStr<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.text.cmp(other.text)
    }
}

impl fmt::Display for GraphemeStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} chars)", self.text, self.grapheme_count)
    }
}

// --- VTable ---

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct GraphemeUtf8ExtVTable;

impl ExtDTypeVTable for GraphemeUtf8ExtVTable {
    type Metadata = GraphemeMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("grapheme_utf8")
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        vortex_ensure!(storage_dtype.is_utf8(), "storage dtype must be UTF-8");
        Ok(())
    }
}

impl ExtScalarVTable for GraphemeUtf8ExtVTable {
    type Value<'a> = GraphemeStr<'a>;

    fn unpack<'a>(
        &self,
        metadata: &'a <Self as ExtDTypeVTable>::Metadata,
        _storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> Self::Value<'a> {
        GraphemeStr {
            text: storage_value.as_utf8().as_ref(),
            grapheme_count: metadata.grapheme_count,
        }
    }

    fn validate_scalar_value(
        &self,
        metadata: &<Self as ExtDTypeVTable>::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        let text: &str = storage_value.as_utf8().as_ref();
        let actual_count = text.chars().count();
        vortex_ensure!(
            actual_count == metadata.grapheme_count,
            "grapheme count mismatch: metadata says {} but string has {}",
            metadata.grapheme_count,
            actual_count
        );
        Ok(())
    }
}

// --- Helpers ---

fn grapheme_scalar(s: &str) -> Scalar {
    let metadata = GraphemeMetadata {
        grapheme_count: s.chars().count(),
    };
    Scalar::extension::<GraphemeUtf8ExtVTable>(metadata, Scalar::utf8(s, Nullability::NonNullable))
}

fn grapheme_metadata(s: &str) -> GraphemeMetadata {
    GraphemeMetadata {
        grapheme_count: s.chars().count(),
    }
}

// --- Tests ---

#[test]
fn test_equality() {
    let s1 = grapheme_scalar("hello");
    let s2 = grapheme_scalar("hello");
    let s3 = grapheme_scalar("world");

    let e1 = s1.as_extension();
    let e2 = s2.as_extension();
    let e3 = s3.as_extension();
    let v1 = e1.as_value::<GraphemeUtf8ExtVTable>();
    let v2 = e2.as_value::<GraphemeUtf8ExtVTable>();
    let v3 = e3.as_value::<GraphemeUtf8ExtVTable>();

    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
}

#[test]
fn test_lexicographic_ordering() {
    let apple = grapheme_scalar("apple");
    let banana = grapheme_scalar("banana");
    let abc = grapheme_scalar("abc");
    let abd = grapheme_scalar("abd");

    let e_apple = apple.as_extension();
    let e_banana = banana.as_extension();
    let e_abc = abc.as_extension();
    let e_abd = abd.as_extension();
    let v_apple = e_apple.as_value::<GraphemeUtf8ExtVTable>();
    let v_banana = e_banana.as_value::<GraphemeUtf8ExtVTable>();
    let v_abc = e_abc.as_value::<GraphemeUtf8ExtVTable>();
    let v_abd = e_abd.as_value::<GraphemeUtf8ExtVTable>();

    assert!(v_apple < v_banana);
    assert!(v_banana > v_apple);
    assert!(v_abc < v_abd);
}

#[test]
fn test_prefix_ordering() {
    // A shorter string is less than a longer one that shares the same prefix.
    let ab = grapheme_scalar("ab");
    let abc = grapheme_scalar("abc");

    let e_ab = ab.as_extension();
    let e_abc = abc.as_extension();
    let v_ab = e_ab.as_value::<GraphemeUtf8ExtVTable>();
    let v_abc = e_abc.as_value::<GraphemeUtf8ExtVTable>();

    assert!(v_ab < v_abc);
}

#[test]
fn test_unicode_ordering() {
    // Multi-byte characters still compare lexicographically by UTF-8 byte ordering.
    let cafe = grapheme_scalar("caf\u{00e9}");
    let caff = grapheme_scalar("caff");

    let e_cafe = cafe.as_extension();
    let e_caff = caff.as_extension();
    let v_cafe = e_cafe.as_value::<GraphemeUtf8ExtVTable>();
    let v_caff = e_caff.as_value::<GraphemeUtf8ExtVTable>();

    // U+00E9 encodes as bytes 0xC3 0xA9, which sorts after 'f' (0x66).
    assert!(v_cafe > v_caff);
}

#[test]
fn test_hash_consistency() {
    use vortex_utils::aliases::hash_set::HashSet;

    let s1 = grapheme_scalar("hello");
    let s2 = grapheme_scalar("hello");

    let mut set = HashSet::new();
    set.insert(s1);
    set.insert(s2);
    assert_eq!(set.len(), 1);

    let s3 = grapheme_scalar("world");
    set.insert(s3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_storage_roundtrip() {
    let storage = Scalar::utf8("hello", Nullability::NonNullable);
    let ext =
        Scalar::extension::<GraphemeUtf8ExtVTable>(grapheme_metadata("hello"), storage.clone());

    assert_eq!(ext.as_extension().to_storage_scalar(), storage);
}

#[test]
fn test_grapheme_count_in_value() {
    // Verify the unpacked value carries the correct grapheme count from metadata.
    let s = grapheme_scalar("caf\u{00e9}");
    let ext = s.as_extension();
    let v = ext.as_value::<GraphemeUtf8ExtVTable>().unwrap();

    assert_eq!(v.grapheme_count, 4);
    assert_eq!(v.text, "caf\u{00e9}");
}

#[test]
fn test_validation_rejects_wrong_grapheme_count() {
    use vortex_dtype::ExtDType;

    let bad_metadata = GraphemeMetadata { grapheme_count: 99 };
    let ext_dtype = ExtDType::try_new(bad_metadata, DType::Utf8(Nullability::NonNullable)).unwrap();

    let result = ExtScalarValue::<GraphemeUtf8ExtVTable>::try_new(
        ext_dtype,
        ScalarValue::Utf8("hello".into()),
    );
    assert!(result.is_err());
}

#[test]
fn test_display() {
    let s = grapheme_scalar("hello");
    let ext = s.as_extension();
    let v = ext.as_value::<GraphemeUtf8ExtVTable>().unwrap();

    assert_eq!(format!("{v}"), "hello (5 chars)");
}
