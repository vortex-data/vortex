// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::FieldName;

/// A path within a variant value.
///
/// Each path element addresses either an object field or a list index.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VariantPath(Vec<VariantPathElement>);

impl VariantPath {
    /// Creates a path from a sequence of path elements.
    pub fn new(path: Vec<VariantPathElement>) -> Self {
        Self(path)
    }

    /// Creates a path that addresses a single object field.
    pub fn from_name(name: impl Into<FieldName>) -> Self {
        Self(vec![VariantPathElement::Field(name.into())])
    }

    /// Creates a path that addresses a single list index.
    pub fn from_index(index: usize) -> Self {
        Self(vec![VariantPathElement::Index(index)])
    }

    /// Returns a new path with an additional element appended.
    pub fn join(mut self, element: impl Into<VariantPathElement>) -> Self {
        self.push(element);
        self
    }

    /// Appends an element to the end of the path.
    pub fn push(&mut self, element: impl Into<VariantPathElement>) {
        self.0.push(element.into());
    }

    /// Iterates over the path elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &VariantPathElement> + '_ {
        self.0.iter()
    }
}

impl<F> From<F> for VariantPath
where
    F: Into<FieldName>,
{
    fn from(value: F) -> Self {
        Self::from_name(value)
    }
}

impl From<usize> for VariantPath {
    fn from(value: usize) -> Self {
        Self::from_index(value)
    }
}

impl std::fmt::Display for VariantPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (idx, element) in self.0.iter().enumerate() {
            match element {
                VariantPathElement::Field(name) if idx == 0 => write!(f, "{name}")?,
                VariantPathElement::Field(name) => write!(f, ".{name}")?,
                VariantPathElement::Index(index) => write!(f, "[{index}]")?,
            }
        }

        Ok(())
    }
}

impl FromIterator<VariantPathElement> for VariantPath {
    fn from_iter<T: IntoIterator<Item = VariantPathElement>>(iter: T) -> Self {
        Self::new(Vec::from_iter(iter))
    }
}

/// A single step within a [`VariantPath`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VariantPathElement {
    /// Select a named field from an object-like variant value.
    Field(FieldName),
    /// Select an element from a list-like variant value.
    Index(usize),
}

impl<F> From<F> for VariantPathElement
where
    F: Into<FieldName>,
{
    fn from(value: F) -> Self {
        Self::Field(value.into())
    }
}

impl From<usize> for VariantPathElement {
    fn from(value: usize) -> Self {
        Self::Index(value)
    }
}

impl VariantPathElement {
    /// Creates a field path element.
    pub fn field(name: impl Into<FieldName>) -> Self {
        Self::Field(name.into())
    }

    /// Creates an index path element.
    pub fn index(index: usize) -> Self {
        Self::Index(index)
    }
}

impl std::fmt::Display for VariantPathElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(name) => write!(f, "{name}"),
            Self::Index(index) => write!(f, "[{index}]"),
        }
    }
}
