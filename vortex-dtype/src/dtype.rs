use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use flatbuffers::{root, root_unchecked};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult, VortexUnwrap,
};
use vortex_flatbuffers::dtype as fbd;
use DType::*;

use crate::field::Field;
use crate::nullability::Nullability;
use crate::{ExtDType, PType};

/// A name for a field in a struct
pub type FieldName = Arc<str>;
/// An ordered list of field names in a struct
pub type FieldNames = Arc<[FieldName]>;

/// The logical types of elements in Vortex arrays.
///
/// Vortex arrays preserve a single logical type, while the encodings allow for multiple
/// physical ways to encode that type.
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// The logical null type (only has a single value, `null`)
    Null,
    /// The logical boolean type (`true` or `false` if non-nullable; `true`, `false`, or `null` if nullable)
    Bool(Nullability),
    /// Primitive, fixed-width numeric types (e.g., `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f32`, `f64`)
    Primitive(PType, Nullability),
    /// UTF-8 strings
    Utf8(Nullability),
    /// Binary data
    Binary(Nullability),
    /// A struct is composed of an ordered list of fields, each with a corresponding name and DType
    /// TODO(ngates): we may want StructDType to be Arc<[Field]> instead so it's only a single Arc.
    Struct(StructDType, Nullability),
    /// A variable-length list type, parameterized by a single element DType
    List(Arc<DType>, Nullability),
    /// User-defined extension types
    Extension(Arc<ExtDType>),
}

impl DType {
    /// The default DType for bytes
    pub const BYTES: Self = Primitive(PType::U8, Nullability::NonNullable);

    /// Get the nullability of the DType
    pub fn nullability(&self) -> Nullability {
        self.is_nullable().into()
    }

    /// Check if the DType is nullable
    pub fn is_nullable(&self) -> bool {
        use crate::nullability::Nullability::*;

        match self {
            Null => true,
            Bool(n) => matches!(n, Nullable),
            Primitive(_, n) => matches!(n, Nullable),
            Utf8(n) => matches!(n, Nullable),
            Binary(n) => matches!(n, Nullable),
            Struct(_, n) => matches!(n, Nullable),
            List(_, n) => matches!(n, Nullable),
            Extension(ext_dtype) => ext_dtype.storage_dtype().is_nullable(),
        }
    }

    /// Get a new DType with `Nullability::NonNullable` (but otherwise the same as `self`)
    pub fn as_nonnullable(&self) -> Self {
        self.with_nullability(Nullability::NonNullable)
    }

    /// Get a new DType with `Nullability::Nullable` (but otherwise the same as `self`)
    pub fn as_nullable(&self) -> Self {
        self.with_nullability(Nullability::Nullable)
    }

    /// Get a new DType with the given nullability (but otherwise the same as `self`)
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        match self {
            Null => Null,
            Bool(_) => Bool(nullability),
            Primitive(p, _) => Primitive(*p, nullability),
            Utf8(_) => Utf8(nullability),
            Binary(_) => Binary(nullability),
            Struct(st, _) => Struct(st.clone(), nullability),
            List(c, _) => List(c.clone(), nullability),
            Extension(ext) => Extension(Arc::new(ext.with_nullability(nullability))),
        }
    }

    /// Check if `self` and `other` are equal, ignoring nullability
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.as_nullable().eq(&other.as_nullable())
    }

    /// Check if `self` is a `StructDType`
    pub fn is_struct(&self) -> bool {
        matches!(self, Struct(_, _))
    }

    /// Check if `self` is an unsigned integer
    pub fn is_unsigned_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_unsigned_int)
    }

    /// Check if `self` is a signed integer
    pub fn is_signed_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_signed_int)
    }

    /// Check if `self` is an integer (signed or unsigned)
    pub fn is_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_int)
    }

    /// Check if `self` is a floating point number
    pub fn is_float(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_float)
    }

    /// Check if `self` is a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, Bool(_))
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
    pub fn as_struct(&self) -> Option<&StructDType> {
        match self {
            Struct(s, _) => Some(s),
            _ => None,
        }
    }

    /// Get the inner dtype if `self` is a `ListDType`, otherwise `None`
    pub fn as_list_element(&self) -> Option<&DType> {
        match self {
            List(s, _) => Some(s.as_ref()),
            _ => None,
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Null => write!(f, "null"),
            Bool(n) => write!(f, "bool{}", n),
            Primitive(pt, n) => write!(f, "{}{}", pt, n),
            Utf8(n) => write!(f, "utf8{}", n),
            Binary(n) => write!(f, "binary{}", n),
            Struct(sdt, n) => write!(
                f,
                "{{{}}}{}",
                sdt.names()
                    .iter()
                    .zip(sdt.dtypes())
                    .map(|(n, dt)| format!("{}={}", n, dt))
                    .join(", "),
                n
            ),
            List(edt, n) => write!(f, "list({}){}", edt, n),
            Extension(ext) => write!(
                f,
                "ext({}, {}{}){}",
                ext.id(),
                ext.storage_dtype()
                    .with_nullability(Nullability::NonNullable),
                ext.metadata()
                    .map(|m| format!(", {:?}", m))
                    .unwrap_or_else(|| "".to_string()),
                ext.storage_dtype().nullability(),
            ),
        }
    }
}

/// A lazily evaluated DType, parsed on access from an underlying flatbuffer.
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq)]
pub struct ViewedDType {
    /// Underlying flatbuffer
    buffer: ByteBuffer,
    /// Location of the dtype data inside the underlying buffer
    flatbuffer_loc: usize,
}

impl ViewedDType {
    /// Create a [`ViewedDType`] from a [`fbd::DType`] and the shared buffer.
    pub(crate) fn from_fb(fb_dtype: fbd::DType<'_>, buffer: ByteBuffer) -> Self {
        Self::with_location(fb_dtype._tab.loc(), buffer)
    }

    /// Create a [`ViewedDType`] from a buffer and a flatbuffer location
    pub(crate) fn with_location(location: usize, buffer: ByteBuffer) -> Self {
        Self {
            buffer,
            flatbuffer_loc: location,
        }
    }

    /// The viewed [`fbd::DType`] instance.
    pub fn flatbuffer(&self) -> fbd::DType<'_> {
        unsafe {
            fbd::DType::init_from_table(flatbuffers::Table::new(
                self.buffer.as_ref(),
                self.flatbuffer_loc,
            ))
        }
    }

    /// Returns the underlying shared buffer
    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }
}

/// DType of a struct's field, either owned or a pointer to an underlying flatbuffer.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldDType {
    inner: FieldDTypeInner,
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
                let owned = DType::try_from(view.clone()).vortex_expect("");
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

    /// Creates a new instance from a flatbuffer-defined object and its underlying buffer.
    pub fn from_fb(fb_struct: fbd::Struct_<'_>, buffer: ByteBuffer) -> VortexResult<Self> {
        let names = fb_struct
            .names()
            .ok_or_else(|| vortex_err!("failed to parse struct names from flatbuffer"))?
            .iter()
            .map(|n| (*n).into())
            .collect_vec()
            .into();

        let dtypes = fb_struct
            .dtypes()
            .ok_or_else(|| vortex_err!("failed to parse struct dtypes from flatbuffer"))?
            .iter()
            .map(|dt| FieldDType {
                inner: FieldDTypeInner::View(ViewedDType::from_fb(dt, buffer.clone())),
            })
            .collect::<Vec<FieldDType>>();

        Ok(StructDType::from_fields(names, dtypes))
    }

    /// Create a new [`StructDType`] from flatbuffer bytes.
    pub fn from_bytes(buffer: ByteBuffer) -> VortexResult<Self> {
        let fb_struct = root::<fbd::DType>(&buffer)?
            .type__as_struct_()
            .ok_or_else(|| vortex_err!("failed to parse struct from flatbuffer"))?;

        Self::from_fb(fb_struct, buffer.clone())
    }

    /// # Safety
    /// Parse a StructDType out of a buffer, must be validated by the other otherwise might panic or behave unexpectedly.
    pub unsafe fn from_bytes_unchecked(buffer: ByteBuffer) -> Self {
        let fb_struct = unsafe { root_unchecked::<fbd::DType>(&buffer) }
            .type__as_struct_()
            .vortex_expect("failed to parse struct from flatbuffer");
        Self::from_fb(fb_struct, buffer.clone()).vortex_expect("Failed to build StructDType")
    }

    /// Get the names of the fields in the struct
    pub fn names(&self) -> &FieldNames {
        &self.names
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

#[cfg(test)]
mod test {
    use std::mem;

    use crate::dtype::DType;
    use crate::field::Field;
    use crate::{Nullability, PType, StructDType};

    #[test]
    fn size_of() {
        assert_eq!(mem::size_of::<DType>(), 40);
    }

    #[test]
    fn nullability() {
        assert!(!DType::Struct(
            StructDType::new(vec![].into(), Vec::new()),
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
            StructDType::new(
                vec!["A".into(), "B".into()].into(),
                vec![a_type.clone(), b_type.clone()],
            ),
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
