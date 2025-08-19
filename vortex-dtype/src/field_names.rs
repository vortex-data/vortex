// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::ops::Index;
use std::sync::Arc;

use itertools::Itertools;

/// A name for a field in a struct.
pub type FieldName = Arc<str>;

/// An ordered list of field names in a struct.
#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldNames(Arc<[FieldName]>);

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
        self.len() == other.len() && self.iter().zip_eq(other.iter()).all(|(l, r)| &**l == *r)
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

impl From<&[&'static str]> for FieldNames {
    fn from(value: &[&'static str]) -> Self {
        Self(value.iter().cloned().map(Arc::from).collect())
    }
}

impl From<&[FieldName]> for FieldNames {
    fn from(value: &[FieldName]) -> Self {
        Self(Arc::from(value))
    }
}

impl<const N: usize> From<[&'static str; N]> for FieldNames {
    fn from(value: [&'static str; N]) -> Self {
        Self(value.into_iter().map(Arc::from).collect())
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
