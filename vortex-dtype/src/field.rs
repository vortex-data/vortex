//! Selectors for fields or elements in (possibly nested) `DType`s
//!
//! A `Field` indexes a single layer of `DType`, for example: a name in a struct or the element of a
//! list. A `FieldPath` indexes zero or more layers, for example: the field "child" which is within
//! the struct field "parent" which is within the struct field "grandparent".

use core::fmt;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::{VortexResult, vortex_bail};

use crate::DType;

/// Selects a nested type within either a struct or a list.
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

impl From<Arc<str>> for Field {
    fn from(value: Arc<str>) -> Self {
        Self::Name(value)
    }
}

impl From<&Arc<str>> for Field {
    fn from(value: &Arc<str>) -> Self {
        Self::Name(value.clone())
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

/// A sequence of field selectors representing a path through zero or more layers of `DType`.
///
/// # Examples
///
/// The empty path references the root:
///
/// ```
/// use vortex_dtype::*;
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

impl FieldPath {
    /// The selector for the root (i.e., the top-level struct itself)
    pub fn root() -> Self {
        Self::default()
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
    /// use vortex_dtype::*;
    /// use vortex_dtype::Nullability::*;
    ///
    /// let dtype = DType::Struct(
    ///     Arc::new(StructFields::from_iter([(
    ///         "a",
    ///         DType::List(
    ///             Arc::new(DType::Struct(
    ///                 Arc::new(StructFields::from_iter([(
    ///                     "b",
    ///                     DType::Primitive(PType::U32, NonNullable),
    ///                 )])),
    ///                 NonNullable,
    ///             )),
    ///             Nullable,
    ///         ),
    ///     )])),
    ///     NonNullable,
    /// );
    ///
    /// let path = FieldPath::from(vec![Field::from("a"), Field::ElementType, Field::from("b")]);
    /// let resolved = path.resolve(dtype).unwrap();
    /// assert_eq!(resolved, DType::Primitive(PType::U32, NonNullable));
    /// ```
    pub fn resolve(&self, mut dtype: DType) -> VortexResult<DType> {
        for field in &self.0 {
            dtype = match (dtype, field) {
                (DType::Struct(fields, _), Field::Name(name)) => fields.field(name)?,
                (DType::List(element_dtype, _), Field::ElementType) => DType::clone(&element_dtype),
                (other, f) => {
                    vortex_bail!("FieldPath: invalid index {:?} for DType {:?}", f, other)
                }
            }
        }

        Ok(dtype)
    }

    /// Does the field referenced by the field path exist in the given dtype?
    pub fn exists(&self, dtype: DType) -> bool {
        // Indexing a struct type always allocates anyway.
        self.resolve(dtype).is_ok()
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
    use crate::Nullability::*;
    use crate::{DType, PType, StructFields};

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

    #[test]
    fn nested_field_single_level() {
        let a_type = DType::Primitive(PType::I32, NonNullable);
        let dtype = DType::Struct(
            Arc::from(StructFields::from_iter([
                ("a", a_type.clone()),
                ("b", DType::Bool(Nullable)),
            ])),
            NonNullable,
        );
        let path = FieldPath::from_name("a");
        assert_eq!(a_type, path.resolve(dtype.clone()).unwrap());
        assert!(path.exists(dtype));
    }

    #[test]
    fn nested_field_two_level() {
        let inner = DType::Struct(
            Arc::new(StructFields::from_iter([
                ("inner_a", DType::Primitive(PType::U8, NonNullable)),
                ("inner_b", DType::Bool(Nullable)),
            ])),
            NonNullable,
        );

        let outer = DType::Struct(
            Arc::from(StructFields::from_iter([
                ("outer_a", DType::Bool(NonNullable)),
                ("outer_b", inner),
            ])),
            NonNullable,
        );

        let path = FieldPath::from_name("outer_b").push("inner_a");
        let dtype = path.resolve(outer.clone()).unwrap();

        assert_eq!(dtype, DType::Primitive(PType::U8, NonNullable));
        assert!(path.exists(outer));
    }

    #[test]
    fn nested_field_deep_nested() {
        let level4 = DType::Struct(
            Arc::new(StructFields::from_iter([(
                "c",
                DType::Primitive(PType::F64, Nullable),
            )])),
            NonNullable,
        );

        let level3 = DType::List(Arc::from(level4), Nullable);

        let level2 = DType::Struct(
            Arc::new(StructFields::from_iter([("b", level3)])),
            NonNullable,
        );

        let level1 = DType::Struct(
            Arc::from(StructFields::from_iter([("a", level2)])),
            NonNullable,
        );

        let path = FieldPath::from_name("a")
            .push("b")
            .push(Field::ElementType)
            .push("c");
        let dtype = path.resolve(level1.clone()).unwrap();

        assert_eq!(dtype, DType::Primitive(PType::F64, Nullable));
        assert!(path.exists(level1.clone()));

        let path = FieldPath::from_name("a")
            .push("b")
            .push("c")
            .push(Field::ElementType);
        assert!(path.resolve(level1.clone()).is_err());
        assert!(!path.exists(level1.clone()));

        let path = FieldPath::from_name("a")
            .push(Field::ElementType)
            .push("b")
            .push("c");
        assert!(path.resolve(level1.clone()).is_err());
        assert!(!path.exists(level1.clone()));

        let path = FieldPath::from_name(Field::ElementType)
            .push("a")
            .push("b")
            .push("c");
        assert!(path.resolve(level1.clone()).is_err());
        assert!(!path.exists(level1));
    }

    #[test]
    fn nested_field_not_found() {
        let dtype = DType::Struct(
            Arc::from(StructFields::from_iter([("a", DType::Bool(NonNullable))])),
            NonNullable,
        );
        let path = FieldPath::from_name("b");
        assert!(path.resolve(dtype.clone()).is_err());
        assert!(!path.exists(dtype.clone()));

        let path = FieldPath::from(Field::ElementType);
        assert!(path.resolve(dtype.clone()).is_err());
        assert!(!path.exists(dtype));
    }

    #[test]
    fn nested_field_non_struct_intermediate() {
        let dtype = DType::Struct(
            Arc::from(StructFields::from_iter([(
                "a",
                DType::Primitive(PType::I32, NonNullable),
            )])),
            NonNullable,
        );
        let path = FieldPath::from_name("a").push("b");
        assert!(path.resolve(dtype.clone()).is_err());
        assert!(!path.exists(dtype.clone()));

        let path = FieldPath::from_name("a").push(Field::ElementType);
        assert!(path.resolve(dtype.clone()).is_err());
        assert!(!path.exists(dtype));
    }
}
