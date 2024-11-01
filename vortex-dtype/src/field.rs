//! Selectors for fields in (possibly nested) `StructDType`s
//!
//! A `Field` can either be a direct child field of the top-level struct (selected by name or index),
//! or a nested field (selected by a sequence of such selectors)

use core::fmt;
use std::fmt::{Display, Formatter};

use itertools::Itertools;

/// A selector for a field in a struct
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Field {
    /// A field selector by name
    Name(String),
    /// A field selector by index (position)
    Index(usize),
}

impl From<&str> for Field {
    fn from(value: &str) -> Self {
        Field::Name(value.into())
    }
}

impl From<String> for Field {
    fn from(value: String) -> Self {
        Field::Name(value)
    }
}

impl From<usize> for Field {
    fn from(value: usize) -> Self {
        Field::Index(value)
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Field::Name(name) => write!(f, "${name}"),
            Field::Index(idx) => write!(f, "[{idx}]"),
        }
    }
}

/// A path through a (possibly nested) struct, composed of a sequence of field selectors
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

    /// Pushes a new field selector to the end of this path
    pub fn push<F: Into<Field>>(&mut self, field: F) {
        self.0.push(field.into());
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
        let mut path = FieldPath::from_name("A");
        path.push("B");
        path.push("C");
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
