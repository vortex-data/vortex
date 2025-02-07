//! Selectors for fields in (possibly nested) `StructDType`s
//!
//! A `Field` can either be a direct child field of the top-level struct (selected by name or index),
//! or a nested field (selected by a sequence of such selectors)

use core::fmt;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use itertools::Itertools;

/// A selector for a field in a struct
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Field {
    /// Address a field of a [`crate::DType::Struct`].
    Name(Arc<str>),
    /// Address the element type of a [`crate::DType::List`].
    ElementType,
}

impl From<&str> for Field {
    fn from(value: &str) -> Self {
        Field::Name(value.into())
    }
}

impl From<String> for Field {
    fn from(value: String) -> Self {
        Field::Name(value.into())
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Field::Name(name) => write!(f, "${name}"),
            Field::ElementType => write!(f, "[]"),
        }
    }
}

impl Field {
    /// Returns true if the field is defined by Name
    pub fn is_named(&self) -> bool {
        matches!(self, Field::Name(_))
    }
}

/// A path through a (possibly nested) struct, composed of a sequence of field selectors
// TODO(ngates): wrap `Vec<Field>` in Option for cheaper "root" path.
// TODO(ngates): we should probably reverse the path. Or better yet, store a Arc<[Field]> along
//  with a positional index to allow cheap step_into.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldPath(Vec<Field>);

impl FieldPath {
    /// The selector for the root (i.e., the top-level struct itself)
    pub fn root() -> Self {
        Self(vec![])
    }

    /// Constructs a new `FieldPath` from a single field selector (i.e., a direct child field of the top-level struct)
    pub fn from_name<F: Into<Field>>(name: F) -> Self {
        Self(vec![name.into()])
    }

    /// Returns the sequence of field selectors that make up this path
    pub fn path(&self) -> &[Field] {
        &self.0
    }

    /// Returns whether this path is a root path.
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Pushes a new field selector to the end of this path
    pub fn push<F: Into<Field>>(mut self, field: F) -> Self {
        self.0.push(field.into());
        self
    }

    /// Whether the path starts with the given field name
    /// TODO(joe): handle asserts better.
    pub fn starts_with_field(&self, field: &Field) -> bool {
        assert!(matches!(field, Field::Name(_)));
        let first = self.0.first();
        assert!(matches!(first, Some(Field::Name(_))));
        first.is_some_and(|f| f == field)
    }

    /// Steps into the next field in the path
    pub fn step_into(mut self) -> Option<Self> {
        if self.0.is_empty() {
            return None;
        }
        self.0 = self.0.iter().skip(1).cloned().collect();
        Some(self)
    }
}

impl FromIterator<Field> for FieldPath {
    fn from_iter<T: IntoIterator<Item = Field>>(iter: T) -> Self {
        FieldPath(iter.into_iter().collect())
    }
}

impl From<Field> for FieldPath {
    fn from(value: Field) -> Self {
        FieldPath(vec![value])
    }
}

impl From<Vec<Field>> for FieldPath {
    fn from(value: Vec<Field>) -> Self {
        FieldPath(value)
    }
}

impl Display for FieldPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0.iter().format("."), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_path() {
        let path = FieldPath::from_name("A").push("B").push("C");
        assert_eq!(path.to_string(), "$A.$B.$C");

        let fields = vec!["A", "B", "C"]
            .into_iter()
            .map(Field::from)
            .collect_vec();
        assert_eq!(path.path(), &fields);

        let vec_path = FieldPath::from(fields);
        assert_eq!(vec_path.to_string(), "$A.$B.$C");
        assert_eq!(path, vec_path);
    }
}
