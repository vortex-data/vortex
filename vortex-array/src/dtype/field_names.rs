// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Borrow;
use std::fmt;
use std::ops::Index;
use std::sync::Arc;

use itertools::Itertools;
use vortex_utils::aliases::StringEscape;

/// A name for a field in a struct.
#[derive(Clone, Debug, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)] // manual PartialEq adds Arc::ptr_eq fast path only
pub struct FieldName(Arc<str>);

impl PartialEq for FieldName {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl FieldName {
    /// Returns a reference to the inner string
    pub fn inner(&self) -> &Arc<str> {
        &self.0
    }
}

// We manually implement serde for `FieldName` so it can round-trip with any string type
#[cfg(feature = "serde")]
impl serde::ser::Serialize for FieldName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::de::Deserialize<'de> for FieldName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: Arc<str> = serde::de::Deserialize::deserialize(deserializer)?;
        Ok(Self::from(s))
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", StringEscape(self.0.as_ref()))
    }
}

impl AsRef<str> for FieldName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for FieldName {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl Borrow<str> for &FieldName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&FieldName> for FieldName {
    fn eq(&self, other: &&FieldName) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialEq<FieldName> for &FieldName {
    fn eq(&self, other: &FieldName) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialEq<&str> for FieldName {
    fn eq(&self, other: &&str) -> bool {
        self.as_ref() == *other
    }
}

impl PartialEq<FieldName> for str {
    fn eq(&self, other: &FieldName) -> bool {
        self == other.as_ref()
    }
}

impl PartialEq<&str> for &FieldName {
    fn eq(&self, other: &&str) -> bool {
        self.as_ref() == *other
    }
}

impl PartialEq<String> for FieldName {
    fn eq(&self, other: &String) -> bool {
        self.as_ref() == other
    }
}

impl PartialEq<&String> for FieldName {
    fn eq(&self, other: &&String) -> bool {
        self.as_ref() == *other
    }
}

impl From<Arc<str>> for FieldName {
    fn from(value: Arc<str>) -> Self {
        Self(value)
    }
}

impl From<&str> for FieldName {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl From<String> for FieldName {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<FieldName> for String {
    fn from(value: FieldName) -> Self {
        value.as_ref().to_string()
    }
}

impl From<FieldName> for Arc<str> {
    fn from(value: FieldName) -> Self {
        value.0
    }
}

/// An ordered list of field names in a struct.
#[derive(Clone, Eq, Debug, Default, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)] // manual PartialEq adds Arc::ptr_eq fast path only
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldNames(Arc<[FieldName]>);

impl PartialEq for FieldNames {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl fmt::Display for FieldNames {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            itertools::join(self.0.iter().map(|n| format!("\"{n}\"")), ", ")
        )
    }
}

impl PartialEq<&FieldNames> for FieldNames {
    fn eq(&self, other: &&FieldNames) -> bool {
        self == *other
    }
}

impl PartialEq<&[&str]> for FieldNames {
    fn eq(&self, other: &&[&str]) -> bool {
        self.len() == other.len() && self.iter().zip_eq(other.iter()).all(|(l, r)| l == r)
    }
}

impl PartialEq<&[&str]> for &FieldNames {
    fn eq(&self, other: &&[&str]) -> bool {
        *self == other
    }
}

impl<const N: usize> PartialEq<[&str; N]> for FieldNames {
    fn eq(&self, other: &[&str; N]) -> bool {
        self == other.as_slice()
    }
}

impl<const N: usize> PartialEq<[&str; N]> for &FieldNames {
    fn eq(&self, other: &[&str; N]) -> bool {
        *self == other.as_slice()
    }
}

impl PartialEq<&[FieldName]> for FieldNames {
    fn eq(&self, other: &&[FieldName]) -> bool {
        self.0.as_ref() == *other
    }
}

impl PartialEq<&[FieldName]> for &FieldNames {
    fn eq(&self, other: &&[FieldName]) -> bool {
        self.0.as_ref() == *other
    }
}

impl FieldNames {
    /// Returns an empty list of names.
    pub fn empty() -> Self {
        Self([].into())
    }
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the number of elements is 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a borrowed iterator over the field names.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &FieldName> {
        FieldNamesIter {
            inner: self,
            idx: 0,
        }
    }

    /// Returns a reference to a field name, or None if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<&FieldName> {
        self.0.get(index)
    }

    /// Finds the index of a field name, or None if not found.
    pub fn find(&self, name: impl AsRef<str>) -> Option<usize> {
        let name_ref = name.as_ref();
        self.iter().position(|n| n.as_ref() == name_ref)
    }
}

impl AsRef<[FieldName]> for FieldNames {
    fn as_ref(&self) -> &[FieldName] {
        &self.0
    }
}

impl Index<usize> for FieldNames {
    type Output = FieldName;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Iterator of references to field names.
pub struct FieldNamesIter<'a> {
    inner: &'a FieldNames,
    idx: usize,
}

impl<'a> Iterator for FieldNamesIter<'a> {
    type Item = &'a FieldName;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.inner.len() {
            return None;
        }

        let i = &self.inner.0[self.idx];
        self.idx += 1;
        Some(i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len() - self.idx;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FieldNamesIter<'_> {}

/// Owned iterator of field names.
pub struct FieldNamesIntoIter {
    inner: FieldNames,
    idx: usize,
}

impl Iterator for FieldNamesIntoIter {
    type Item = FieldName;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.inner.len() {
            return None;
        }

        let i = self.inner.0[self.idx].clone();
        self.idx += 1;
        Some(i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len() - self.idx;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FieldNamesIntoIter {}

impl IntoIterator for FieldNames {
    type Item = FieldName;

    type IntoIter = FieldNamesIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        FieldNamesIntoIter {
            inner: self,
            idx: 0,
        }
    }
}

impl From<Vec<FieldName>> for FieldNames {
    fn from(value: Vec<FieldName>) -> Self {
        Self(value.into())
    }
}

impl From<Vec<Arc<str>>> for FieldNames {
    fn from(value: Vec<Arc<str>>) -> Self {
        value.into_iter().collect()
    }
}

impl From<&[&'static str]> for FieldNames {
    fn from(value: &[&'static str]) -> Self {
        Self(value.iter().cloned().map(FieldName::from).collect())
    }
}

impl<const N: usize> From<[&str; N]> for FieldNames {
    fn from(value: [&str; N]) -> Self {
        Self(value.iter().cloned().map(FieldName::from).collect())
    }
}

impl From<Vec<&str>> for FieldNames {
    fn from(value: Vec<&str>) -> Self {
        Self(value.into_iter().map(FieldName::from).collect())
    }
}

impl From<&[FieldName]> for FieldNames {
    fn from(value: &[FieldName]) -> Self {
        Self(Arc::from(value))
    }
}

impl<const N: usize> From<&[&str; N]> for FieldNames {
    fn from(value: &[&str; N]) -> Self {
        Self(value.iter().cloned().map(FieldName::from).collect())
    }
}

impl<const N: usize> From<[FieldName; N]> for FieldNames {
    fn from(value: [FieldName; N]) -> Self {
        Self(value.into())
    }
}

impl<F: Into<FieldName>> FromIterator<F> for FieldNames {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self(iter.into_iter().map(|v| v.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_names_iter() {
        let names = ["a", "b"];
        let field_names = FieldNames::from(names);
        assert_eq!(field_names.iter().len(), names.len());
        let mut iter = field_names.iter();
        assert_eq!(iter.next(), Some(&"a".into()));
        assert_eq!(iter.next(), Some(&"b".into()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_field_names_owned_iter() {
        let names = ["a", "b"];
        let field_names = FieldNames::from(names);
        assert_eq!(field_names.clone().into_iter().len(), names.len());
        let mut iter = field_names.into_iter();
        assert_eq!(iter.next(), Some("a".into()));
        assert_eq!(iter.next(), Some("b".into()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_field_names_equality() {
        let field_names = FieldNames::from(["field1", "field2", "field3"]);

        // FieldNames == &FieldNames
        let field_names_ref = &field_names;
        assert_eq!(field_names, field_names_ref);

        // FieldNames == &[&str]
        let str_slice = &["field1", "field2", "field3"][..];
        assert_eq!(field_names, str_slice);

        // &FieldNames == &[&str]
        assert_eq!(&field_names, str_slice);

        // FieldNames == [&str; N] (array)
        assert_eq!(field_names, ["field1", "field2", "field3"]);

        // &FieldNames == [&str; N] (array)
        assert_eq!(&field_names, ["field1", "field2", "field3"]);

        // FieldNames == &[FieldName]
        let field_name_vec: Vec<FieldName> =
            vec!["field1".into(), "field2".into(), "field3".into()];
        let field_name_slice = field_name_vec.as_slice();
        assert_eq!(field_names, field_name_slice);

        // &FieldNames == &[FieldName]
        assert_eq!(&field_names, field_name_slice);

        // Test inequality cases
        assert_ne!(field_names, &["field1", "field2"][..]);
        assert_ne!(field_names, ["different", "fields", "here"]);
        assert_ne!(field_names, &["field1", "field2", "field3", "extra"][..]);
    }

    #[test]
    fn test_field_names_display() {
        let names = FieldNames::from(["a", "b", "c"]);
        let f = format!("{names}");

        assert_eq!(f, r#"["a", "b", "c"]"#);
    }

    /// Tests both that contains is correct but also that the types are set up correctly.
    #[test]
    fn test_field_names_contains() {
        let names = FieldNames::from(["a", "b", "c"]);
        assert!(names.iter().contains("b"))
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_field_name_serde() {
        let s = "hello world";
        let value = serde_json::to_value(s).unwrap();
        let name = serde_json::from_value::<FieldName>(value).unwrap();
        assert_eq!(name, s);
        let value = serde_json::to_value(name.clone()).unwrap();
        let s = serde_json::from_value::<String>(value).unwrap();
        assert_eq!(name, s);
    }

    /// Verify hashing is unchanged and behaves as expected
    #[test]
    fn test_hash_behavior() {
        use std::hash::BuildHasher;

        let rs = std::hash::RandomState::new();
        assert_eq!(rs.hash_one("hello"), rs.hash_one(FieldName::from("hello")));
    }
}
