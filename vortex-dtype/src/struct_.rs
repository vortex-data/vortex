use std::hash::Hash;
use std::sync::Arc;

use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult, VortexUnwrap,
};

use crate::flatbuffers::ViewedDType;
use crate::{DType, Field, FieldName, FieldNames};

/// DType of a struct's field, either owned or a pointer to an underlying flatbuffer.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Hash)]
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

#[derive(Debug, Clone, Eq)]
enum FieldDTypeInner {
    /// Owned DType instance
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

impl PartialOrd for FieldDTypeInner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (FieldDTypeInner::Owned(lhs), FieldDTypeInner::Owned(rhs)) => lhs.partial_cmp(rhs),
            (FieldDTypeInner::View(lhs), FieldDTypeInner::View(rhs)) => {
                let lhs = DType::try_from(lhs.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");
                let rhs = DType::try_from(rhs.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");

                lhs.partial_cmp(&rhs)
            }
            (FieldDTypeInner::Owned(dtype), FieldDTypeInner::View(viewed_dtype)) => {
                let rhs = DType::try_from(viewed_dtype.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");

                dtype.partial_cmp(&rhs)
            }
            (FieldDTypeInner::View(viewed_dtype), FieldDTypeInner::Owned(dtype)) => {
                let lhs = DType::try_from(viewed_dtype.clone())
                    .vortex_expect("Failed to parse FieldDType into DType");

                lhs.partial_cmp(dtype)
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

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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

/// A struct dtype is a list of names and corresponding dtypes
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructDType {
    names: FieldNames,
    dtypes: Arc<[FieldDType]>,
}

/// Information about a field in a struct dtype
#[derive(Debug)]
pub struct FieldInfo {
    /// The position index of the field within the enclosing struct
    pub index: usize,
    /// The name of the field
    pub name: Arc<str>,
    /// The dtype of the field
    pub dtype: FieldDType,
}

impl StructDType {
    /// Create a new [`StructDType`] from a list of names and dtypes
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
            .collect::<Vec<_>>()
            .into();

        Self { names, dtypes }
    }

    /// Create a new [`StructDType`] from a  list of names and [`FieldDType`] which can be either lazily or eagerly serialized.
    pub fn from_fields(names: FieldNames, dtypes: Vec<FieldDType>) -> Self {
        if names.len() != dtypes.len() {
            vortex_panic!(
                "length mismatch between names ({}) and dtypes ({})",
                names.len(),
                dtypes.len()
            );
        }

        Self {
            names,
            dtypes: dtypes.into(),
        }
    }

    /// Get the names of the fields in the struct
    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    /// Find the index of a field.
    pub fn find(&self, field: &Field) -> Option<usize> {
        match field {
            Field::Name(name) => self.find_name(name),
            Field::Index(idx) => Some(*idx),
        }
    }

    /// Find the index of a field by name
    /// Returns `None` if the field is not found
    pub fn find_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_ref() == name)
    }

    /// Get information about the referenced field, either by name or index
    /// Returns an error if the field is not found
    pub fn field_info(&self, field: &Field) -> VortexResult<FieldInfo> {
        let index = match field {
            Field::Name(name) => self
                .find_name(name)
                .ok_or_else(|| vortex_err!("Unknown field: {name}"))?,
            Field::Index(index) => *index,
        };
        if index >= self.names.len() {
            vortex_bail!("field index out of bounds: {index}")
        }
        Ok(FieldInfo {
            index,
            name: self.names[index].clone(),
            dtype: self.dtypes[index].clone(),
        })
    }

    /// Get the type of specific field by index
    pub fn field_dtype(&self, index: usize) -> VortexResult<DType> {
        self.dtypes[index].value()
    }

    /// Returns an ordered iterator over the members of Self.
    pub fn dtypes(&self) -> impl ExactSizeIterator<Item = DType> + '_ {
        self.dtypes.iter().map(|dt| dt.value().vortex_unwrap())
    }

    /// Project a subset of fields from the struct
    /// Returns an error if any of the referenced fields are not found
    pub fn project(&self, projection: &[Field]) -> VortexResult<Self> {
        let mut names = Vec::with_capacity(projection.len());
        let mut dtypes = Vec::with_capacity(projection.len());

        for field in projection.iter() {
            let FieldInfo { name, dtype, .. } = self.field_info(field)?;

            names.push(name.clone());
            dtypes.push(dtype.clone());
        }

        Ok(StructDType::from_fields(names.into(), dtypes))
    }
}

impl<T, V> FromIterator<(T, V)> for StructDType
where
    T: Into<FieldName>,
    V: Into<FieldDType>,
{
    fn from_iter<I: IntoIterator<Item = (T, V)>>(iter: I) -> Self {
        let (names, dtypes): (Vec<_>, Vec<_>) = iter
            .into_iter()
            .map(|(name, dtype)| (name.into(), dtype.into()))
            .unzip();
        StructDType::from_fields(names.into(), dtypes.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod test {
    use crate::dtype::DType;
    use crate::field::Field;
    use crate::{Nullability, PType, StructDType};

    #[test]
    fn nullability() {
        assert!(!DType::Struct(
            StructDType::new(vec![].into(), Vec::new()).into(),
            Nullability::NonNullable
        )
        .is_nullable());

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
            StructDType::from_iter([("A", a_type.clone()), ("B", b_type.clone())]).into(),
            Nullability::Nullable,
        );
        assert!(dtype.is_nullable());
        assert!(dtype.as_struct().is_some());
        assert!(a_type.as_struct().is_none());

        let sdt = dtype.as_struct().unwrap();
        assert_eq!(sdt.names().len(), 2);
        assert_eq!(sdt.dtypes().len(), 2);
        assert_eq!(sdt.names()[0], "A".into());
        assert_eq!(sdt.names()[1], "B".into());
        assert_eq!(sdt.field_dtype(0).unwrap(), a_type);
        assert_eq!(sdt.field_dtype(1).unwrap(), b_type);

        let proj = sdt
            .project(&[Field::Index(1), Field::Name("A".into())])
            .unwrap();
        assert_eq!(proj.names()[0], "B".into());
        assert_eq!(proj.field_dtype(0).unwrap(), b_type);
        assert_eq!(proj.names()[1], "A".into());
        assert_eq!(proj.field_dtype(1).unwrap(), a_type);

        let field_info = sdt.field_info(&Field::Name("B".into())).unwrap();
        assert_eq!(field_info.index, 1);
        assert_eq!(field_info.name, "B".into());
        assert_eq!(field_info.dtype.value().unwrap(), b_type);

        let field_info = sdt.field_info(&Field::Index(0)).unwrap();
        assert_eq!(field_info.index, 0);
        assert_eq!(field_info.name, "A".into());
        assert_eq!(field_info.dtype.value().unwrap(), a_type);

        assert!(sdt.field_info(&Field::Index(2)).is_err());

        assert_eq!(sdt.find_name("A"), Some(0));
        assert_eq!(sdt.find_name("B"), Some(1));
        assert_eq!(sdt.find_name("C"), None);
    }
}
