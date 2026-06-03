// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selectors for fields or elements in (possibly nested) `DType`s
//!
//! A `Field` indexes a single layer of `DType`, for example: a name in a struct or the element of a
//! list. A `FieldPath` indexes zero or more layers, for example: the field "child" which is within
//! the struct field "parent" which is within the struct field "grandparent".

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::DType;
use crate::dtype::FieldName;

/// Selects a nested type within either a struct or a list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Field {
    /// Address a field of a [`crate::dtype::DType::Struct`].
    Name(FieldName),
    // TODO(connor)[FixedSizeList]: Actually make use of this variant after `FixedSizeList` is
    // implemented.
    /// Address the element type of a [`crate::dtype::DType::List`] or [`crate::dtype::DType::FixedSizeList`].
    ElementType,
}

impl Field {
    /// Retrieve a field name if it has one
    pub fn as_name(&self) -> Option<&str> {
        match self {
            Field::Name(name) => Some(name.as_ref()),
            Field::ElementType => None,
        }
    }

    /// Returns true if the field is defined by Name
    pub fn is_named(&self) -> bool {
        matches!(self, Field::Name(_))
    }
}

impl From<&str> for Field {
    fn from(value: &str) -> Self {
        Field::Name(value.into())
    }
}

impl From<Arc<str>> for Field {
    fn from(value: Arc<str>) -> Self {
        Self::Name(FieldName::from(value))
    }
}

impl From<FieldName> for Field {
    fn from(value: FieldName) -> Self {
        Self::Name(value)
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

/// A sequence of field selectors representing a path through zero or more layers of `DType`.
///
/// # Examples
///
/// The empty path references the root:
///
/// ```
/// use vortex_array::dtype::*;
///
/// let dtype_i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
/// assert_eq!(dtype_i32, FieldPath::root().resolve(dtype_i32.clone()).unwrap());
/// ```
///
// TODO(ngates): we should probably reverse the path. Or better yet, store a Arc<[Field]> along
//  with a positional index to allow cheap step_into.
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldPath(Vec<Field>);

/// A helpful constructor for creating `FieldPath`s to nested
/// struct fields of the format `field_path!(x.y.z)`
#[macro_export]
macro_rules! field_path {
    ($front:ident) => {{
        $crate::dtype::FieldPath::from_name(stringify!($front))
    }};
    ($front:ident $(. $rest:ident)+) => {{
        $crate::dtype::FieldPath::from_iter([
            $crate::dtype::Field::from(stringify!($front)),
            $($crate::dtype::Field::from(stringify!($rest))),+
        ])
    }};
}

impl FieldPath {
    /// The selector for the root (i.e., the top-level struct itself)
    pub fn root() -> Self {
        Self::default()
    }

    /// Constructs a new `FieldPath` from a single field selector (i.e., a direct child field of the top-level struct)
    pub fn from_name<F: Into<FieldName>>(name: F) -> Self {
        Self(vec![Field::Name(name.into())])
    }

    /// Returns the sequence of field selectors that make up this path
    pub fn parts(&self) -> &[Field] {
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
        self.0 = self.0.into_iter().skip(1).collect();
        Some(self)
    }

    /// The dtype, within the given type, to which this field path refers.
    ///
    /// Note that a nullable DType may contain a non-nullable DType. This function returns the
    /// literal nullability of the child.
    ///
    /// # Examples
    ///
    /// Extract the type of the "b" field from `struct{a: list(struct{b: u32})?}`:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use vortex_array::dtype::*;
    /// use vortex_array::dtype::Nullability::*;
    ///
    /// let dtype = DType::Struct(
    ///     StructFields::from_iter([(
    ///         "a",
    ///         DType::List(
    ///             Arc::new(DType::Struct(
    ///                 StructFields::from_iter([(
    ///                     "b",
    ///                     DType::Primitive(PType::U32, NonNullable),
    ///                 )]),
    ///                 NonNullable,
    ///             )),
    ///             Nullable,
    ///         ),
    ///     )]),
    ///     NonNullable,
    /// );
    ///
    /// let path = FieldPath::from(vec![Field::from("a"), Field::ElementType, Field::from("b")]);
    /// let resolved = path.resolve(dtype).unwrap();
    /// assert_eq!(resolved, DType::Primitive(PType::U32, NonNullable));
    /// ```
    pub fn resolve(&self, mut dtype: DType) -> Option<DType> {
        for field in &self.0 {
            dtype = match (dtype, field) {
                (DType::Struct(fields, _), Field::Name(name)) => fields.field(name)?,
                (DType::List(element_dtype, _), Field::ElementType) => {
                    element_dtype.as_ref().clone()
                }
                (..) => return None,
            }
        }

        Some(dtype)
    }

    /// Does the field referenced by the field path exist in the given dtype?
    pub fn exists_in(&self, dtype: DType) -> bool {
        // Indexing a struct type always allocates anyway.
        self.resolve(dtype).is_some()
    }

    /// Returns true if this path overlaps with another path (i.e., one is a prefix of the other).
    pub fn overlap(&self, other: &FieldPath) -> bool {
        let min_len = self.0.len().min(other.0.len());
        self.0.iter().take(min_len).eq(other.0.iter().take(min_len))
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

#[derive(Default, Clone, Debug)]
/// A set of field paths supporting efficient `contains` queries.
///
/// Paths are stored as inserted. Prefix-minimization—collapsing a path into an ancestor that
/// already covers it—is deferred until the set is iterated via [`IntoIterator`], so insertion stays
/// cheap.
pub struct FieldPathSet {
    /// While this is currently a set wrapper it can be replaced with a trie, at which point the
    /// deferred minimization in [`IntoIterator`] becomes cheap.
    // TODO(joe): this can be replaced with a `FieldPath` trie
    set: HashSet<FieldPath>,
}

impl FieldPathSet {
    /// Checks if the set contains exactly this field path.
    pub fn contains(&self, path: &FieldPath) -> bool {
        self.set.contains(path)
    }

    /// Iterates over the field paths in the set, as inserted (not prefix-minimized).
    pub fn iter(&self) -> impl Iterator<Item = &FieldPath> {
        self.set.iter()
    }

    /// Inserts a field path. Prefix-minimization is deferred until the set is iterated.
    pub fn insert(&mut self, path: FieldPath) {
        self.set.insert(path);
    }
}

/// Reduces field paths to their minimal covering set: any path that has another path in the set as
/// a prefix is redundant and dropped.
fn minimal_covering_set(paths: impl IntoIterator<Item = FieldPath>) -> Vec<FieldPath> {
    let mut covering: Vec<FieldPath> = Vec::new();
    for path in paths {
        if covering
            .iter()
            .any(|existing| path.parts().starts_with(existing.parts()))
        {
            continue;
        }
        covering.retain(|existing| !existing.parts().starts_with(path.parts()));
        covering.push(path);
    }
    covering
}

impl Extend<FieldPath> for FieldPathSet {
    fn extend<T: IntoIterator<Item = FieldPath>>(&mut self, iter: T) {
        self.set.extend(iter);
    }
}

impl FromIterator<FieldPath> for FieldPathSet {
    fn from_iter<T: IntoIterator<Item = FieldPath>>(iter: T) -> Self {
        let set = HashSet::from_iter(iter);
        Self { set }
    }
}

impl IntoIterator for FieldPathSet {
    type Item = FieldPath;
    type IntoIter = std::vec::IntoIter<FieldPath>;

    /// Iterates the prefix-minimal covering set: redundant descendants are dropped.
    fn into_iter(self) -> Self::IntoIter {
        minimal_covering_set(self.set).into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::*;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

    #[test]
    fn test_field_path() {
        let path = FieldPath::from_name("A").push("B").push("C");
        assert_eq!(path.to_string(), "$A.$B.$C");

        let fields = vec!["A", "B", "C"]
            .into_iter()
            .map(Field::from)
            .collect_vec();
        assert_eq!(path.parts(), &fields);

        let vec_path = FieldPath::from(fields);
        assert_eq!(vec_path.to_string(), "$A.$B.$C");
        assert_eq!(path, vec_path);
    }

    #[test]
    fn nested_field_single_level() {
        let a_type = DType::Primitive(PType::I32, NonNullable);
        let dtype = DType::struct_(
            [("a", a_type.clone()), ("b", DType::Bool(Nullable))],
            NonNullable,
        );
        let path = FieldPath::from_name("a");
        assert_eq!(a_type, path.resolve(dtype.clone()).unwrap());
        assert!(path.exists_in(dtype));
    }

    #[test]
    fn nested_field_two_level() {
        let inner = DType::struct_(
            [
                ("inner_a", DType::Primitive(PType::U8, NonNullable)),
                ("inner_b", DType::Bool(Nullable)),
            ],
            NonNullable,
        );

        let outer = DType::Struct(
            StructFields::from_iter([("outer_a", DType::Bool(NonNullable)), ("outer_b", inner)]),
            NonNullable,
        );

        let path = FieldPath::from_name("outer_b").push("inner_a");
        let dtype = path.resolve(outer.clone()).unwrap();

        assert_eq!(dtype, DType::Primitive(PType::U8, NonNullable));
        assert!(path.exists_in(outer));
    }

    #[test]
    fn nested_field_deep_nested() {
        let level1 = DType::struct_(
            [(
                "a",
                DType::struct_(
                    [(
                        "b",
                        DType::list(
                            DType::struct_(
                                [("c", DType::Primitive(PType::F64, Nullable))],
                                NonNullable,
                            ),
                            Nullable,
                        ),
                    )],
                    NonNullable,
                ),
            )],
            NonNullable,
        );

        let path = FieldPath::from_name("a")
            .push("b")
            .push(Field::ElementType)
            .push("c");
        let dtype = path.resolve(level1.clone()).unwrap();

        assert_eq!(dtype, DType::Primitive(PType::F64, Nullable));
        assert!(path.exists_in(level1.clone()));

        let path = FieldPath::from_name("a")
            .push("b")
            .push("c")
            .push(Field::ElementType);
        assert!(path.resolve(level1.clone()).is_none());
        assert!(!path.exists_in(level1.clone()));

        let path = FieldPath::from_name("a")
            .push(Field::ElementType)
            .push("b")
            .push("c");
        assert!(path.resolve(level1.clone()).is_none());
        assert!(!path.exists_in(level1.clone()));

        let path = FieldPath::root().push("a").push("b").push("c");
        assert!(path.resolve(level1.clone()).is_none());
        assert!(!path.exists_in(level1));
    }

    #[test]
    fn nested_field_not_found() {
        let dtype = DType::struct_([("a", DType::Bool(NonNullable))], NonNullable);
        let path = field_path!(b);
        assert!(path.resolve(dtype.clone()).is_none());
        assert!(!path.exists_in(dtype.clone()));

        let path = FieldPath::from(Field::ElementType);
        assert!(path.resolve(dtype.clone()).is_none());
        assert!(!path.exists_in(dtype));
    }

    #[test]
    fn nested_field_non_struct_intermediate() {
        let dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, NonNullable))],
            NonNullable,
        );
        let path = field_path!(a.b);
        assert!(path.resolve(dtype.clone()).is_none());
        assert!(!path.exists_in(dtype.clone()));

        let path = FieldPath::from_name("a").push(Field::ElementType);
        assert!(path.resolve(dtype.clone()).is_none());
        assert!(!path.exists_in(dtype));
    }

    #[test]
    fn test_overlap_positive() {
        let path1 = field_path!(a.b.c);
        let path2 = field_path!(a.b);
        assert!(path1.overlap(&path2));
        assert!(path2.overlap(&path1));

        let path3 = field_path!(a);
        assert!(path1.overlap(&path3));
        assert!(path3.overlap(&path1));
    }

    #[test]
    fn test_overlap_negative() {
        let path1 = field_path!(a.b.c);
        let path2 = field_path!(a.x.y);
        assert!(!path1.overlap(&path2));
        assert!(!path2.overlap(&path1));

        let path3 = field_path!(x);
        assert!(!path1.overlap(&path3));
        assert!(!path3.overlap(&path1));
    }

    #[test]
    fn iteration_yields_minimal_covering_set() {
        let mut paths = FieldPathSet::default();
        paths.extend([field_path!(a.b), field_path!(x), field_path!(a)]);
        paths.insert(field_path!(a.c));

        // Iteration collapses `a.b`/`a.c` into the covering `a`.
        assert_eq!(
            paths.into_iter().collect::<HashSet<_>>(),
            HashSet::from_iter([field_path!(a), field_path!(x)])
        );
    }
}
