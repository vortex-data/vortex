// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_error::{
    VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_err, vortex_panic,
};
use vortex_flatbuffers::{FlatBufferRoot, WriteFlatBuffer};

use crate::flatbuffers::ViewedDType;
use crate::{DType, FieldName, FieldNames, PType};
use crate::{Nullability, flatbuffers as fb};

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
            (Self::View(lhs), Self::View(rhs)) => Self::viewed_eq(lhs, rhs),
            (Self::View(view), Self::Owned(owned)) | (Self::Owned(owned), Self::View(view)) => {
                Self::owned_vs_viewed_eq(owned, view)
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
                let viewed_fb = view.flatbuffer();

                match viewed_fb.type_type() {
                    fb::Type::Null => {
                        // Hash as DType::Null
                        std::mem::discriminant(&DType::Null).hash(state);
                    }
                    fb::Type::Bool => {
                        let bool_fb = viewed_fb
                            .type__as_bool()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(bool_fb.nullable());
                        let dtype = DType::Bool(nullability);
                        std::mem::discriminant(&dtype).hash(state);
                        nullability.hash(state);
                    }
                    fb::Type::Primitive => {
                        let prim_fb = viewed_fb
                            .type__as_primitive()
                            .vortex_expect("must be valid flatbuffer");
                        if let Ok(ptype) = PType::try_from(prim_fb.ptype()) {
                            let nullability = Nullability::from(prim_fb.nullable());
                            let dtype = DType::Primitive(ptype, nullability);
                            std::mem::discriminant(&dtype).hash(state);
                            ptype.hash(state);
                            nullability.hash(state);
                        }
                    }
                    fb::Type::Decimal => {
                        let dec_fb = viewed_fb
                            .type__as_decimal()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(dec_fb.nullable());
                        if let Ok(decimal_dtype) =
                            crate::DecimalDType::try_new(dec_fb.precision(), dec_fb.scale())
                        {
                            let dtype = DType::Decimal(decimal_dtype, nullability);
                            std::mem::discriminant(&dtype).hash(state);
                            decimal_dtype.hash(state);
                            nullability.hash(state);
                        }
                    }
                    fb::Type::Binary => {
                        let bin_fb = viewed_fb
                            .type__as_binary()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(bin_fb.nullable());
                        let dtype = DType::Binary(nullability);
                        std::mem::discriminant(&dtype).hash(state);
                        nullability.hash(state);
                    }
                    fb::Type::Utf8 => {
                        let utf8_fb = viewed_fb
                            .type__as_utf_8()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(utf8_fb.nullable());
                        let dtype = DType::Utf8(nullability);
                        std::mem::discriminant(&dtype).hash(state);
                        nullability.hash(state);
                    }
                    // For complex types, fall back to parsing for now
                    _ => {
                        if let Ok(owned_dt) = DType::try_from(view.clone()) {
                            owned_dt.hash(state);
                        }
                    }
                }
            }
        }
    }
}

impl FieldDType {
    /// Returns the concrete DType, parsing it from the underlying buffer if necessary.
    pub fn value(&self) -> VortexResult<DType> {
        self.inner.value()
    }

    /// Compare two FieldDTypes ignoring nullability without allocating.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.inner.eq_ignore_nullability(&other.inner)
    }

    /// Check if the FieldDType is nullable without allocating.
    pub fn is_nullable(&self) -> bool {
        match &self.inner {
            FieldDTypeInner::Owned(owned) => owned.is_nullable(),
            FieldDTypeInner::View(view) => {
                let viewed_fb = view.flatbuffer();
                match viewed_fb.type_type() {
                    fb::Type::Null => true,
                    fb::Type::Bool => viewed_fb
                        .type__as_bool()
                        .vortex_expect("must be valid flatbuffer")
                        .nullable(),
                    fb::Type::Primitive => viewed_fb
                        .type__as_primitive()
                        .vortex_expect("must be valid flatbuffer")
                        .nullable(),
                    fb::Type::Decimal => viewed_fb
                        .type__as_decimal()
                        .vortex_expect("must be valid flatbuffer")
                        .nullable(),
                    fb::Type::Binary => viewed_fb
                        .type__as_binary()
                        .vortex_expect("must be valid flatbuffer")
                        .nullable(),
                    fb::Type::Utf8 => viewed_fb
                        .type__as_utf_8()
                        .vortex_expect("must be valid flatbuffer")
                        .nullable(),
                    // For complex types, fall back to parsing
                    _ => self.value().map(|dt| dt.is_nullable()).unwrap_or(false),
                }
            }
        }
    }

    /// Convert to Arrow DataType.
    #[cfg(feature = "arrow")]
    pub fn to_arrow_dtype(&self) -> VortexResult<arrow_schema::DataType> {
        self.value()?.to_arrow_dtype()
    }
}

impl FlatBufferRoot for FieldDType {}

impl WriteFlatBuffer for FieldDType {
    type Target<'a> = crate::flatbuffers::DType<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        match &self.inner {
            FieldDTypeInner::Owned(owned) => owned.write_flatbuffer(fbb),
            FieldDTypeInner::View(view) => view.write_flatbuffer(fbb),
        }
    }
}

impl PartialEq<DType> for FieldDType {
    fn eq(&self, other: &DType) -> bool {
        match &self.inner {
            FieldDTypeInner::Owned(owned) => owned == other,
            FieldDTypeInner::View(view) => FieldDTypeInner::owned_vs_viewed_eq(other, view),
        }
    }
}

impl PartialEq<&DType> for FieldDType {
    fn eq(&self, other: &&DType) -> bool {
        match &self.inner {
            FieldDTypeInner::Owned(owned) => owned.eq(*other),
            FieldDTypeInner::View(view) => FieldDTypeInner::owned_vs_viewed_eq(other, view),
        }
    }
}

impl PartialEq<FieldDType> for DType {
    fn eq(&self, other: &FieldDType) -> bool {
        match &other.inner {
            FieldDTypeInner::Owned(dtype) => self == dtype,
            FieldDTypeInner::View(view) => FieldDTypeInner::owned_vs_viewed_eq(self, view),
        }
    }
}

impl PartialEq<&FieldDType> for DType {
    fn eq(&self, other: &&FieldDType) -> bool {
        match &other.inner {
            FieldDTypeInner::Owned(dtype) => self == dtype,
            FieldDTypeInner::View(view) => FieldDTypeInner::owned_vs_viewed_eq(self, view),
        }
    }
}

impl PartialEq<DType> for &FieldDType {
    fn eq(&self, other: &DType) -> bool {
        (*self).eq(other)
    }
}

impl Display for FieldDType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            FieldDTypeInner::Owned(owned) => write!(f, "{}", owned),
            FieldDTypeInner::View(view) => {
                // For simple types, format directly without allocating DType
                let viewed_fb = view.flatbuffer();
                match viewed_fb.type_type() {
                    fb::Type::Null => write!(f, "null"),
                    fb::Type::Bool => {
                        let bool_fb = viewed_fb
                            .type__as_bool()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(bool_fb.nullable());
                        write!(f, "bool{}", nullability)
                    }
                    fb::Type::Primitive => {
                        let prim_fb = viewed_fb
                            .type__as_primitive()
                            .vortex_expect("must be valid flatbuffer");
                        if let Ok(ptype) = PType::try_from(prim_fb.ptype()) {
                            let nullability = Nullability::from(prim_fb.nullable());
                            write!(f, "{}{}", ptype, nullability)
                        } else {
                            write!(f, "unknown_primitive")
                        }
                    }
                    fb::Type::Decimal => {
                        let dec_fb = viewed_fb
                            .type__as_decimal()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(dec_fb.nullable());
                        if let Ok(decimal_dtype) =
                            crate::DecimalDType::try_new(dec_fb.precision(), dec_fb.scale())
                        {
                            write!(f, "{}{}", decimal_dtype, nullability)
                        } else {
                            write!(f, "invalid_decimal")
                        }
                    }
                    fb::Type::Binary => {
                        let bin_fb = viewed_fb
                            .type__as_binary()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(bin_fb.nullable());
                        write!(f, "binary{}", nullability)
                    }
                    fb::Type::Utf8 => {
                        let utf8_fb = viewed_fb
                            .type__as_utf_8()
                            .vortex_expect("must be valid flatbuffer");
                        let nullability = Nullability::from(utf8_fb.nullable());
                        write!(f, "utf8{}", nullability)
                    }
                    // For complex types, fall back to DType conversion
                    _ => match self.value() {
                        Ok(dtype) => write!(f, "{}", dtype),
                        Err(_) => write!(f, "invalid_dtype"),
                    },
                }
            }
        }
    }
}

impl FieldDTypeInner {
    #[inline]
    fn value(&self) -> VortexResult<DType> {
        match &self {
            FieldDTypeInner::Owned(owned) => Ok(owned.clone()),
            FieldDTypeInner::View(view) => DType::try_from(view.clone()),
        }
    }

    /// Compare two FieldDTypes ignoring nullability without allocating.
    fn eq_ignore_nullability(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(lhs), Self::Owned(rhs)) => lhs.eq_ignore_nullability(rhs),
            (Self::View(lhs), Self::View(rhs)) => Self::viewed_eq_ignore_nullability(lhs, rhs),
            (Self::View(view), Self::Owned(owned)) | (Self::Owned(owned), Self::View(view)) => {
                Self::owned_vs_viewed_eq_ignore_nullability(owned, view)
            }
        }
    }

    /// Compare two ViewedDTypes for equality without allocating DType instances.
    fn viewed_eq(lhs: &ViewedDType, rhs: &ViewedDType) -> bool {
        let lhs_fb = lhs.flatbuffer();
        let rhs_fb = rhs.flatbuffer();

        if lhs_fb.type_type() != rhs_fb.type_type() {
            return false;
        }

        match lhs_fb.type_type() {
            fb::Type::Null => true,
            fb::Type::Bool => {
                let lhs_bool = lhs_fb
                    .type__as_bool()
                    .vortex_expect("must be valid flatbuffer");
                let rhs_bool = rhs_fb
                    .type__as_bool()
                    .vortex_expect("must be valid flatbuffer");
                lhs_bool.nullable() == rhs_bool.nullable()
            }
            fb::Type::Primitive => {
                let lhs_prim = lhs_fb
                    .type__as_primitive()
                    .vortex_expect("must be valid flatbuffer");
                let rhs_prim = rhs_fb
                    .type__as_primitive()
                    .vortex_expect("must be valid flatbuffer");
                lhs_prim.ptype() == rhs_prim.ptype() && lhs_prim.nullable() == rhs_prim.nullable()
            }
            fb::Type::Decimal => {
                let lhs_dec = lhs_fb
                    .type__as_decimal()
                    .vortex_expect("must be valid flatbuffer");
                let rhs_dec = rhs_fb
                    .type__as_decimal()
                    .vortex_expect("must be valid flatbuffer");
                lhs_dec.precision() == rhs_dec.precision()
                    && lhs_dec.scale() == rhs_dec.scale()
                    && lhs_dec.nullable() == rhs_dec.nullable()
            }
            fb::Type::Binary => {
                let lhs_bin = lhs_fb
                    .type__as_binary()
                    .vortex_expect("must be valid flatbuffer");
                let rhs_bin = rhs_fb
                    .type__as_binary()
                    .vortex_expect("must be valid flatbuffer");
                lhs_bin.nullable() == rhs_bin.nullable()
            }
            fb::Type::Utf8 => {
                let lhs_utf8 = lhs_fb
                    .type__as_utf_8()
                    .vortex_expect("must be valid flatbuffer");
                let rhs_utf8 = rhs_fb
                    .type__as_utf_8()
                    .vortex_expect("must be valid flatbuffer");
                lhs_utf8.nullable() == rhs_utf8.nullable()
            }
            // For complex types (List, FixedSizeList, Struct, Extension),
            // fall back to parsing for now to maintain correctness
            _ => {
                // Fall back to allocation-based comparison for complex types
                match (DType::try_from(lhs.clone()), DType::try_from(rhs.clone())) {
                    (Ok(lhs_dt), Ok(rhs_dt)) => lhs_dt == rhs_dt,
                    _ => unreachable!("Both viewed dtypes must by DType buffers"),
                }
            }
        }
    }

    /// Compare ViewedDType with owned DType for equality without extra allocations.
    fn owned_vs_viewed_eq(owned: &DType, viewed: &ViewedDType) -> bool {
        let viewed_fb = viewed.flatbuffer();

        match (owned, viewed_fb.type_type()) {
            (DType::Null, fb::Type::Null) => true,
            (DType::Bool(owned_null), fb::Type::Bool) => {
                let viewed_bool = viewed_fb
                    .type__as_bool()
                    .vortex_expect("must be valid flatbuffer");
                owned_null.is_nullable() == viewed_bool.nullable()
            }
            (DType::Primitive(owned_ptype, owned_null), fb::Type::Primitive) => {
                let viewed_prim = viewed_fb
                    .type__as_primitive()
                    .vortex_expect("must be valid flatbuffer");
                *owned_ptype == viewed_prim.ptype().try_into().unwrap_or(PType::U8)
                    && owned_null.is_nullable() == viewed_prim.nullable()
            }
            (DType::Decimal(owned_dec, owned_null), fb::Type::Decimal) => {
                let viewed_dec = viewed_fb
                    .type__as_decimal()
                    .vortex_expect("must be valid flatbuffer");
                owned_dec.precision() == viewed_dec.precision()
                    && owned_dec.scale() == viewed_dec.scale()
                    && owned_null.is_nullable() == viewed_dec.nullable()
            }
            (DType::Binary(owned_null), fb::Type::Binary) => {
                let viewed_bin = viewed_fb
                    .type__as_binary()
                    .vortex_expect("must be valid flatbuffer");
                owned_null.is_nullable() == viewed_bin.nullable()
            }
            (DType::Utf8(owned_null), fb::Type::Utf8) => {
                let viewed_utf8 = viewed_fb
                    .type__as_utf_8()
                    .vortex_expect("must be valid flatbuffer");
                owned_null.is_nullable() == viewed_utf8.nullable()
            }
            // For complex types, fall back to parsing
            _ => match DType::try_from(viewed.clone()) {
                Ok(viewed_dt) => owned == &viewed_dt,
                Err(_) => false,
            },
        }
    }

    /// Compare two ViewedDTypes ignoring nullability without allocating.
    fn viewed_eq_ignore_nullability(lhs: &ViewedDType, rhs: &ViewedDType) -> bool {
        use crate::flatbuffers as fb;

        let lhs_fb = lhs.flatbuffer();
        let rhs_fb = rhs.flatbuffer();

        if lhs_fb.type_type() != rhs_fb.type_type() {
            return false;
        }

        match lhs_fb.type_type() {
            fb::Type::Null => true,
            fb::Type::Bool => true, // Only type matters, not nullability
            fb::Type::Primitive => {
                let lhs_prim = lhs_fb
                    .type__as_primitive()
                    .vortex_expect("valid flatbuffer");
                let rhs_prim = rhs_fb
                    .type__as_primitive()
                    .vortex_expect("valid flatbuffer");
                lhs_prim.ptype() == rhs_prim.ptype()
            }
            fb::Type::Decimal => {
                let lhs_dec = lhs_fb.type__as_decimal().vortex_expect("valid flatbuffer");
                let rhs_dec = rhs_fb.type__as_decimal().vortex_expect("valid flatbuffer");
                lhs_dec.precision() == rhs_dec.precision() && lhs_dec.scale() == rhs_dec.scale()
            }
            fb::Type::Binary | fb::Type::Utf8 => true,
            // For complex types, fall back to parsing for now
            _ => match (DType::try_from(lhs.clone()), DType::try_from(rhs.clone())) {
                (Ok(lhs_dt), Ok(rhs_dt)) => lhs_dt.eq_ignore_nullability(&rhs_dt),
                _ => false,
            },
        }
    }

    /// Compare owned DType with ViewedDType ignoring nullability.
    fn owned_vs_viewed_eq_ignore_nullability(owned: &DType, viewed: &ViewedDType) -> bool {
        let viewed_fb = viewed.flatbuffer();

        match (owned, viewed_fb.type_type()) {
            (DType::Null, fb::Type::Null) => true,
            (DType::Bool(_), fb::Type::Bool) => true, // Ignore nullability
            (DType::Primitive(owned_ptype, _), fb::Type::Primitive) => {
                let viewed_prim = viewed_fb
                    .type__as_primitive()
                    .vortex_expect("valid flatbuffer");
                *owned_ptype == viewed_prim.ptype().try_into().unwrap_or(PType::U8)
                // Ignore nullability
            }
            (DType::Decimal(owned_dec, _), fb::Type::Decimal) => {
                let viewed_dec = viewed_fb
                    .type__as_decimal()
                    .vortex_expect("valid flatbuffer");
                owned_dec.precision() == viewed_dec.precision()
                    && owned_dec.scale() == viewed_dec.scale()
                // Ignore nullability
            }
            (DType::Binary(_), fb::Type::Binary) => true,
            (DType::Utf8(_), fb::Type::Utf8) => true,
            // For complex types, fall back to parsing
            _ => match DType::try_from(viewed.clone()) {
                Ok(viewed_dt) => owned.eq_ignore_nullability(&viewed_dt),
                Err(_) => false,
            },
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
    pub fn fields(&self) -> impl ExactSizeIterator<Item = &FieldDType> + '_ {
        self.0.dtypes.iter()
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
mod tests {
    use itertools::Itertools;
    use vortex_utils::aliases::hash_map::HashMap;

    use crate::dtype::DType;
    use crate::serde::flatbuffers::ViewedDType;
    use crate::{FieldDType, FieldNames, Nullability, PType, StructFields};
    use flatbuffers::root;
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use vortex_flatbuffers::FlatBuffer;
    use vortex_flatbuffers::WriteFlatBufferExt;

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
        assert!(dtype.as_struct_fields_opt().is_some());
        assert!(a_type.as_struct_fields_opt().is_none());

        let sdt = dtype.as_struct_fields_opt().unwrap();
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

    #[test]
    fn test_field_dtype_equality() {
        // Test owned vs owned
        let owned1 = FieldDType::from(DType::Primitive(PType::I32, Nullability::Nullable));
        let owned2 = FieldDType::from(DType::Primitive(PType::I32, Nullability::Nullable));
        let owned3 = FieldDType::from(DType::Primitive(PType::I64, Nullability::Nullable));

        assert_eq!(owned1, owned2);
        assert_ne!(owned1, owned3);

        // Test eq_ignore_nullability
        let nullable = FieldDType::from(DType::Primitive(PType::I32, Nullability::Nullable));
        let non_nullable = FieldDType::from(DType::Primitive(PType::I32, Nullability::NonNullable));

        assert_ne!(nullable, non_nullable);
        assert!(nullable.eq_ignore_nullability(&non_nullable));
    }

    #[test]
    fn test_field_dtype_viewed_equality() {
        // Create a DType and serialize it to flatbuffer
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let bytes = dtype.write_flatbuffer_bytes();
        let root_fb = root::<crate::flatbuffers::DType>(&bytes).unwrap();
        let view = ViewedDType::from_fb_loc(root_fb._tab.loc(), FlatBuffer::from(bytes));

        let owned_field = FieldDType::from(dtype.clone());
        let viewed_field = FieldDType::from(view);

        // Test equality between owned and viewed
        assert_eq!(owned_field, viewed_field);
        assert_eq!(viewed_field, owned_field);

        // Test hash consistency
        let mut map = HashMap::new();
        map.insert(owned_field.clone(), "value");
        assert!(map.contains_key(&viewed_field));

        // Test explicit hash equality

        let build_hasher = RandomState::new();
        let hash1 = build_hasher.hash_one(&owned_field);
        let hash2 = build_hasher.hash_one(&viewed_field);

        assert_eq!(
            hash1, hash2,
            "Hash values must be equal for equal FieldDTypes"
        );

        // Test eq_ignore_nullability
        let non_null_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let non_null_field = FieldDType::from(non_null_dtype);

        assert_ne!(owned_field, non_null_field);
        assert!(owned_field.eq_ignore_nullability(&non_null_field));
        assert!(viewed_field.eq_ignore_nullability(&non_null_field));
    }
}
