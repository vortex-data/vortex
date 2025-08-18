// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::{
    VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_err, vortex_panic,
};

use crate::flatbuffers::ViewedDType;
use crate::{DType, FieldName, FieldNames, PType};

/// DType of a struct's field, either owned or a pointer to an underlying flatbuffer.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldDType {
    inner: FieldDTypeInner,
}

impl From<ViewedDType> for FieldDType {
    fn from(value: ViewedDType) -> Self {
        Self {
            inner: FieldDTypeInner::View(value),
        }
    }
}

impl From<DType> for FieldDType {
    fn from(value: DType) -> Self {
        Self {
            inner: FieldDTypeInner::Owned(value),
        }
    }
}

impl From<PType> for FieldDType {
    fn from(value: PType) -> Self {
        Self {
            inner: FieldDTypeInner::Owned(DType::from(value)),
        }
    }
}

#[derive(Debug, Clone, Eq)]
enum FieldDTypeInner {
    /// Owned DType instance
    // TODO(ngates): we should consider making this an Arc<DType>.
    Owned(DType),
    /// A view over a flatbuffer, parsed only when accessed.
    View(ViewedDType),
}

impl PartialEq for FieldDTypeInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(lhs), Self::Owned(rhs)) => lhs == rhs,
            (Self::View(lhs), Self::View(rhs)) => {
                let lhs = DType::try_from(lhs.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");
                let rhs = DType::try_from(rhs.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");

                lhs == rhs
            }
            (Self::View(view), Self::Owned(owned)) | (Self::Owned(owned), Self::View(view)) => {
                let view = DType::try_from(view.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");
                owned == &view
            }
        }
    }
}

impl Hash for FieldDTypeInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            FieldDTypeInner::Owned(owned) => {
                owned.hash(state);
            }
            FieldDTypeInner::View(view) => {
                let owned = DType::try_from(view.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");
                owned.hash(state);
            }
        }
    }
}

impl FieldDType {
    /// Returns the concrete DType, parsing it from the underlying buffer if necessary.
    pub fn value(&self) -> VortexResult<DType> {
        self.inner.value()
    }
}

impl FieldDTypeInner {
    fn value(&self) -> VortexResult<DType> {
        match &self {
            FieldDTypeInner::Owned(owned) => Ok(owned.clone()),
            FieldDTypeInner::View(view) => DType::try_from(view.clone()),
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for FieldDTypeInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;

        let value = self.value().map_err(S::Error::custom)?;
        serializer.serialize_newtype_variant("FieldDType", 0, "Owned", &value)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FieldDTypeInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_enum("FieldDType", &["Owned", "View"], FieldDTypeDeVisitor)
    }
}

#[cfg(feature = "serde")]
struct FieldDTypeDeVisitor;

#[cfg(feature = "serde")]
impl<'de> serde::de::Visitor<'de> for FieldDTypeDeVisitor {
    type Value = FieldDTypeInner;

    fn expecting(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "variant identifier")
    }

    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::EnumAccess<'de>,
    {
        use serde::de::{Error, VariantAccess};

        #[derive(serde::Deserialize, Debug)]
        enum FieldDTypeVariant {
            Owned,
            View,
        }
        let (variant, variant_data): (FieldDTypeVariant, _) = data.variant()?;

        match variant {
            FieldDTypeVariant::Owned => {
                let inner = variant_data.newtype_variant::<DType>()?;
                Ok(FieldDTypeInner::Owned(inner))
            }
            other => Err(A::Error::custom(format!("unsupported variant {other:?}"))),
        }
    }
}

/// Type information for a struct column.
///
/// The `StructFields` holds all field names and field types, and provides
/// access to them by index or by name.
///
/// ## Duplicate field names
///
/// In memory, it is not an error for a `StructFields` to contain duplicate
/// field names. In that case, any name-based access to fields will resolve
/// to the first such field with a given name.
///
/// ```rust
/// # use vortex_dtype::{DType, Nullability, PType, StructFields};
///
/// let fields = StructFields::from_iter([
///     ("string_col", DType::Utf8(Nullability::NonNullable)),
///     ("binary_col", DType::Binary(Nullability::NonNullable)),
///     ("int_col", DType::Primitive(PType::I32, Nullability::Nullable)),
///     ("int_col", DType::Primitive(PType::I64, Nullability::Nullable)),
/// ]);
///
/// // Accessing a field by name will yield the first
/// assert_eq!(fields.field("int_col").unwrap(), DType::Primitive(PType::I32, Nullability::Nullable));
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructFields(Arc<StructFieldsInner>);

impl std::fmt::Debug for StructFields {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructFields")
            .field("names", &self.0.names)
            .field("dtypes", &self.0.dtypes)
            .finish()
    }
}

impl Display for StructFields {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.names()
                .iter()
                .zip(self.fields())
                .map(|(n, dt)| format!("{n}={dt}"))
                .join(", ")
        )
    }
}

#[derive(PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct StructFieldsInner {
    names: FieldNames,
    dtypes: Arc<[FieldDType]>,
}

impl Default for StructFields {
    fn default() -> Self {
        Self::empty()
    }
}

impl StructFields {
    /// The fields of the empty struct.
    pub fn empty() -> Self {
        Self(Arc::new(StructFieldsInner {
            names: FieldNames::default(),
            dtypes: Arc::from([]),
        }))
    }

    /// Create a new [`StructFields`] from a list of names and dtypes
    pub fn new(names: FieldNames, dtypes: Vec<DType>) -> Self {
        if names.len() != dtypes.len() {
            vortex_panic!(
                "length mismatch between names ({}) and dtypes ({})",
                names.len(),
                dtypes.len()
            );
        }

        let dtypes = dtypes
            .into_iter()
            .map(|dt| FieldDType {
                inner: FieldDTypeInner::Owned(dt),
            })
            .collect::<Vec<_>>();

        Self::from_fields(names, dtypes)
    }

    /// Create a new [`StructFields`] from a  list of names and [`FieldDType`] which can be either lazily or eagerly serialized.
    pub fn from_fields(names: FieldNames, dtypes: Vec<FieldDType>) -> Self {
        if names.len() != dtypes.len() {
            vortex_panic!(
                "length mismatch between names ({}) and dtypes ({})",
                names.len(),
                dtypes.len()
            );
        }

        let inner = Arc::new(StructFieldsInner {
            names,
            dtypes: dtypes.into(),
        });

        Self(inner)
    }

    /// Get the names of the fields in the struct
    pub fn names(&self) -> &FieldNames {
        &self.0.names
    }

    /// Returns the number of fields in the struct
    pub fn nfields(&self) -> usize {
        self.0.names.len()
    }

    /// Returns the name of the field at the given index
    pub fn field_name(&self, index: usize) -> Option<&FieldName> {
        self.0.names.get(index)
    }

    /// Find the index of a field by name
    /// Returns `None` if the field is not found
    pub fn find(&self, name: impl AsRef<str>) -> Option<usize> {
        let name = name.as_ref();
        self.0.names.iter().position(|n| n.as_ref() == name)
    }

    /// Get the [`DType`] of a field.
    ///
    /// It is possible for there to be more than one field with
    /// the same name, in which case, this will return the DType
    /// of the first field encountered with a given name.
    pub fn field(&self, name: impl AsRef<str>) -> Option<DType> {
        let index = self.find(name)?;
        Some(self.0.dtypes[index].value().vortex_unwrap())
    }

    /// Get the [`DType`] of a field by index.
    pub fn field_by_index(&self, index: usize) -> Option<DType> {
        Some(self.0.dtypes.get(index)?.value().vortex_unwrap())
    }

    /// Returns an ordered iterator over the fields.
    pub fn fields(&self) -> impl ExactSizeIterator<Item = DType> + '_ {
        self.0.dtypes.iter().map(|dt| dt.value().vortex_unwrap())
    }

    /// Project a subset of fields from the struct
    ///
    /// If any of the fields are not found, this method will return
    /// an error.
    pub fn project(&self, projection: &[FieldName]) -> VortexResult<Self> {
        let mut names = Vec::with_capacity(projection.len());
        let mut dtypes = Vec::with_capacity(projection.len());

        for field in projection {
            let idx = self
                .find(field)
                .ok_or_else(|| vortex_err!("{field} not found"))?;
            names.push(self.0.names[idx].clone());
            dtypes.push(self.0.dtypes[idx].clone());
        }

        Ok(StructFields::from_fields(names.into(), dtypes))
    }

    /// Returns a new [`StructFields`] without the field at the given index.
    ///
    /// ## Errors
    /// Returns an error if the index is out of bounds for the struct fields.
    pub fn without_field(&self, index: usize) -> VortexResult<Self> {
        if index >= self.nfields() {
            vortex_bail!(
                "index {} out of bounds for struct with {} fields",
                index,
                self.nfields()
            );
        }

        let names = self
            .0
            .names
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != index)
            .map(|(_, name)| name.clone())
            .collect::<FieldNames>();

        let dtypes = self
            .0
            .dtypes
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != index)
            .map(|(_, dtype)| dtype.clone())
            .collect::<Vec<_>>();

        Ok(StructFields::from_fields(names, dtypes))
    }

    /// Merge two [`StructFields`] instances into a new one.
    /// Order of fields in arguments is preserved
    ///
    /// # Errors
    /// Returns an error if the merged struct would have duplicate field names.
    pub fn disjoint_merge(&self, other: &Self) -> VortexResult<Self> {
        let names = self
            .0
            .names
            .iter()
            .chain(other.0.names.iter())
            .cloned()
            .collect::<FieldNames>();

        if !names.iter().all_unique() {
            vortex_bail!("Can't merge struct fields with duplicate names");
        }

        let dtypes = self
            .0
            .dtypes
            .iter()
            .chain(other.0.dtypes.iter())
            .cloned()
            .collect::<Vec<_>>();

        Ok(Self::from_fields(names, dtypes))
    }
}

impl<T, V> FromIterator<(T, V)> for StructFields
where
    T: Into<FieldName>,
    V: Into<FieldDType>,
{
    fn from_iter<I: IntoIterator<Item = (T, V)>>(iter: I) -> Self {
        let (names, dtypes): (Vec<_>, Vec<_>) = iter
            .into_iter()
            .map(|(name, dtype)| (name.into(), dtype.into()))
            .unzip();
        StructFields::from_fields(names.into(), dtypes)
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::dtype::DType;
    use crate::{FieldNames, Nullability, PType, StructFields};

    #[test]
    fn nullability() {
        assert!(
            !DType::Struct(
                StructFields::new(FieldNames::default(), Vec::new()),
                Nullability::NonNullable
            )
            .is_nullable()
        );

        let primitive = DType::Primitive(PType::U8, Nullability::Nullable);
        assert!(primitive.is_nullable());
        assert!(!primitive.as_nonnullable().is_nullable());
        assert!(primitive.as_nonnullable().as_nullable().is_nullable());
    }

    #[test]
    fn test_struct() {
        let a_type = DType::Primitive(PType::I32, Nullability::Nullable);
        let b_type = DType::Bool(Nullability::NonNullable);

        let dtype = DType::Struct(
            StructFields::from_iter([("A", a_type.clone()), ("B", b_type.clone())]),
            Nullability::Nullable,
        );
        assert!(dtype.is_nullable());
        assert!(dtype.as_struct().is_some());
        assert!(a_type.as_struct().is_none());

        let sdt = dtype.as_struct().unwrap();
        assert_eq!(sdt.names().len(), 2);
        assert_eq!(sdt.fields().len(), 2);
        assert_eq!(sdt.names(), ["A", "B"]);
        assert_eq!(sdt.field_by_index(0).unwrap(), a_type);
        assert_eq!(sdt.field_by_index(1).unwrap(), b_type);

        let proj = sdt.project(&["B".into(), "A".into()]).unwrap();
        assert_eq!(proj.names(), ["B", "A"]);
        assert_eq!(proj.field_by_index(0).unwrap(), b_type);
        assert_eq!(proj.field_by_index(1).unwrap(), a_type);

        assert_eq!(sdt.find("A").unwrap(), 0);
        assert_eq!(sdt.find("B").unwrap(), 1);
        assert!(sdt.find("C").is_none());

        let without_a = sdt.without_field(0).unwrap();
        assert_eq!(without_a.names(), ["B"]);
        assert_eq!(without_a.field_by_index(0).unwrap(), b_type);
        assert_eq!(without_a.nfields(), 1);
    }

    #[test]
    fn test_without_field_out_of_bounds() {
        let a_type = DType::Primitive(PType::I32, Nullability::Nullable);
        let b_type = DType::Bool(Nullability::NonNullable);
        let sdt = StructFields::from_iter([("A", a_type), ("B", b_type)]);

        let result = sdt.without_field(2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));

        let result = sdt.without_field(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_without_field_deprecated() {
        let a_type = DType::Primitive(PType::I32, Nullability::Nullable);
        let b_type = DType::Bool(Nullability::NonNullable);
        let sdt = StructFields::from_iter([("A", a_type), ("B", b_type.clone())]);

        let without_a = sdt.without_field(0).unwrap();
        assert_eq!(without_a.names(), ["B"]);
        assert_eq!(without_a.field_by_index(0).unwrap(), b_type);
        assert_eq!(without_a.nfields(), 1);
    }

    #[test]
    fn test_merge() {
        let child_a = DType::Primitive(PType::I32, Nullability::NonNullable);
        let child_b = DType::Bool(Nullability::Nullable);
        let child_c = DType::Utf8(Nullability::NonNullable);

        let sf1 = StructFields::from_iter([("A", child_a.clone()), ("B", child_b.clone())]);

        let sf2 = StructFields::from_iter([("C", child_c.clone())]);

        let merged = StructFields::disjoint_merge(&sf1, &sf2).unwrap();
        assert_eq!(merged.names(), ["A", "B", "C"]);
        assert_eq!(
            merged.fields().collect_vec(),
            vec![child_a, child_b, child_c]
        );

        let err = StructFields::disjoint_merge(&sf1, &sf1).err().unwrap();
        assert!(err.to_string().contains("duplicate names"),);
    }

    #[test]
    fn test_display() {
        let fields = StructFields::from_iter([
            ("name", DType::Utf8(Nullability::NonNullable)),
            ("age", DType::Primitive(PType::I32, Nullability::Nullable)),
            ("active", DType::Bool(Nullability::NonNullable)),
        ]);

        assert_eq!(fields.to_string(), "{name=utf8, age=i32?, active=bool}");

        // Test empty struct
        let empty = StructFields::empty();
        assert_eq!(empty.to_string(), "{}");

        // Test nested struct
        let nested = StructFields::from_iter([
            ("id", DType::Primitive(PType::U64, Nullability::NonNullable)),
            ("data", DType::Struct(fields, Nullability::Nullable)),
        ]);
        assert_eq!(
            nested.to_string(),
            "{id=u64, data={name=utf8, age=i32?, active=bool}?}"
        );
    }
}
