// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde::de;
use serde::de::DeserializeSeed;
use serde::de::EnumAccess;
use serde::de::MapAccess;
use serde::de::SeqAccess;
use serde::de::VariantAccess;
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use serde::ser::SerializeTupleVariant;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::decimal::DecimalDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ForeignExtDType;
use crate::dtype::session::DTypeSessionExt;

/// Serialize Nullability as a boolean
impl Serialize for Nullability {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        bool::from(*self).serialize(serializer)
    }
}

/// Deserialize Nullability from a boolean
impl<'de> Deserialize<'de> for Nullability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        bool::deserialize(deserializer).map(Self::from)
    }
}

/// Seed for deserializing DType references that require session context.
pub struct DTypeSerde<'a, T> {
    session: &'a VortexSession,
    _marker: PhantomData<T>,
}

impl<'a, T> DTypeSerde<'a, T> {
    /// Create a new DTypeSerde seed.
    pub fn new(session: &'a VortexSession) -> Self {
        Self {
            session,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// DType Serialization (tuple variants to match derive format)
// ============================================================================

impl Serialize for DType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DType::Null => serializer.serialize_unit_variant("DType", 0, "Null"),
            DType::Bool(n) => serializer.serialize_newtype_variant("DType", 1, "Bool", n),
            DType::Primitive(ptype, n) => {
                let mut state = serializer.serialize_tuple_variant("DType", 2, "Primitive", 2)?;
                state.serialize_field(ptype)?;
                state.serialize_field(n)?;
                state.end()
            }
            DType::Decimal(decimal, n) => {
                let mut state = serializer.serialize_tuple_variant("DType", 3, "Decimal", 2)?;
                state.serialize_field(decimal)?;
                state.serialize_field(n)?;
                state.end()
            }
            DType::Utf8(n) => serializer.serialize_newtype_variant("DType", 4, "Utf8", n),
            DType::Binary(n) => serializer.serialize_newtype_variant("DType", 5, "Binary", n),
            DType::List(element_dtype, n) => {
                let mut state = serializer.serialize_tuple_variant("DType", 6, "List", 2)?;
                state.serialize_field(element_dtype.as_ref())?;
                state.serialize_field(n)?;
                state.end()
            }
            DType::FixedSizeList(element_dtype, size, n) => {
                let mut state =
                    serializer.serialize_tuple_variant("DType", 7, "FixedSizeList", 3)?;
                state.serialize_field(element_dtype.as_ref())?;
                state.serialize_field(size)?;
                state.serialize_field(n)?;
                state.end()
            }
            DType::Struct(fields, n) => {
                let mut state = serializer.serialize_tuple_variant("DType", 8, "Struct", 2)?;
                state.serialize_field(&fields)?;
                state.serialize_field(n)?;
                state.end()
            }
            DType::Union(n) => serializer.serialize_newtype_variant("DType", 11, "Union", n),
            DType::Extension(ext) => {
                serializer.serialize_newtype_variant("DType", 9, "Extension", ext)
            }
            DType::Variant(n) => serializer.serialize_newtype_variant("DType", 10, "Variant", n),
        }
    }
}

impl Serialize for StructFields {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("StructFields", 2)?;
        state.serialize_field("names", self.names())?;
        let dtypes: Vec<DType> = self.fields().collect();
        state.serialize_field("dtypes", &dtypes)?;
        state.end()
    }
}

// ============================================================================
// DType Deserialization with session context (DeserializeSeed)
// ============================================================================

impl<'de> DeserializeSeed<'de> for DTypeSerde<'_, DType> {
    type Value = DType;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const VARIANTS: &[&str] = &[
            "Null",
            "Bool",
            "Primitive",
            "Decimal",
            "Utf8",
            "Binary",
            "List",
            "FixedSizeList",
            "Struct",
            "Union",
            "Extension",
            "Variant",
        ];

        struct DTypeVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for DTypeVisitor<'_> {
            type Value = DType;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("enum DType")
            }

            fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
            where
                A: EnumAccess<'de>,
            {
                let (variant, access) = data.variant::<Cow<'_, str>>()?;
                match variant.as_ref() {
                    "Null" => {
                        access.unit_variant()?;
                        Ok(DType::Null)
                    }
                    "Bool" => {
                        let n = access.newtype_variant()?;
                        Ok(DType::Bool(n))
                    }
                    "Primitive" => {
                        #[derive(Deserialize)]
                        struct Fields(PType, Nullability);
                        let Fields(ptype, n) = access.newtype_variant()?;
                        Ok(DType::Primitive(ptype, n))
                    }
                    "Decimal" => {
                        #[derive(Deserialize)]
                        struct Fields(DecimalDType, Nullability);
                        let Fields(decimal, n) = access.newtype_variant()?;
                        Ok(DType::Decimal(decimal, n))
                    }
                    "Utf8" => {
                        let n = access.newtype_variant()?;
                        Ok(DType::Utf8(n))
                    }
                    "Binary" => {
                        let n = access.newtype_variant()?;
                        Ok(DType::Binary(n))
                    }
                    "List" => access.newtype_variant_seed(ListFieldsSeed {
                        session: self.session,
                    }),
                    "FixedSizeList" => access.newtype_variant_seed(FixedSizeListFieldsSeed {
                        session: self.session,
                    }),
                    "Struct" => access.newtype_variant_seed(StructFieldsSeed {
                        session: self.session,
                    }),
                    "Union" => {
                        let n = access.newtype_variant()?;
                        Ok(DType::Union(n))
                    }
                    "Extension" => {
                        let ext = access
                            .newtype_variant_seed(DTypeSerde::<ExtDTypeRef>::new(self.session))?;
                        Ok(DType::Extension(ext))
                    }
                    "Variant" => {
                        let n = access.newtype_variant()?;
                        Ok(DType::Variant(n))
                    }
                    _ => Err(de::Error::unknown_variant(&variant, VARIANTS)),
                }
            }
        }

        deserializer.deserialize_enum(
            "DType",
            VARIANTS,
            DTypeVisitor {
                session: self.session,
            },
        )
    }
}

// ============================================================================
// Helper seeds for nested DType variants (with session)
// ============================================================================

struct ListFieldsSeed<'a> {
    session: &'a VortexSession,
}

impl<'de> DeserializeSeed<'de> for ListFieldsSeed<'_> {
    type Value = DType;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ListVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for ListVisitor<'_> {
            type Value = DType;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("List tuple (element_dtype, nullability)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let element_dtype = seq
                    .next_element_seed(DTypeSerde::<DType>::new(self.session))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let nullability = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(DType::List(Arc::new(element_dtype), nullability))
            }
        }

        deserializer.deserialize_tuple(
            2,
            ListVisitor {
                session: self.session,
            },
        )
    }
}

struct FixedSizeListFieldsSeed<'a> {
    session: &'a VortexSession,
}

impl<'de> DeserializeSeed<'de> for FixedSizeListFieldsSeed<'_> {
    type Value = DType;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FixedSizeListVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for FixedSizeListVisitor<'_> {
            type Value = DType;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("FixedSizeList tuple (element_dtype, size, nullability)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let element_dtype = seq
                    .next_element_seed(DTypeSerde::<DType>::new(self.session))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let size = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let nullability = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                Ok(DType::FixedSizeList(
                    Arc::new(element_dtype),
                    size,
                    nullability,
                ))
            }
        }

        deserializer.deserialize_tuple(
            3,
            FixedSizeListVisitor {
                session: self.session,
            },
        )
    }
}

struct StructFieldsSeed<'a> {
    session: &'a VortexSession,
}

impl<'de> DeserializeSeed<'de> for StructFieldsSeed<'_> {
    type Value = DType;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StructVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for StructVisitor<'_> {
            type Value = DType;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("Struct tuple (fields, nullability)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let fields = seq
                    .next_element_seed(DTypeSerde::<StructFields>::new(self.session))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let nullability = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(DType::Struct(fields, nullability))
            }
        }

        deserializer.deserialize_tuple(
            2,
            StructVisitor {
                session: self.session,
            },
        )
    }
}

impl<'de> DeserializeSeed<'de> for DTypeSerde<'_, StructFields> {
    type Value = StructFields;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["names", "dtypes"];

        struct StructFieldsInnerVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for StructFieldsInnerVisitor<'_> {
            type Value = StructFields;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("struct StructFields")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut names: Option<FieldNames> = None;
                let mut dtypes: Option<Vec<DType>> = None;

                while let Some(key) = map.next_key::<Cow<'_, str>>()? {
                    match key.as_ref() {
                        "names" => {
                            if names.is_some() {
                                return Err(de::Error::duplicate_field("names"));
                            }
                            names = Some(map.next_value()?);
                        }
                        "dtypes" => {
                            if dtypes.is_some() {
                                return Err(de::Error::duplicate_field("dtypes"));
                            }
                            dtypes = Some(
                                map.next_value_seed(DTypeSerde::<Vec<DType>>::new(self.session))?,
                            );
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }

                let names = names.ok_or_else(|| de::Error::missing_field("names"))?;
                let dtypes = dtypes.ok_or_else(|| de::Error::missing_field("dtypes"))?;

                Ok(StructFields::new(names, dtypes))
            }
        }

        deserializer.deserialize_struct(
            "StructFields",
            FIELDS,
            StructFieldsInnerVisitor {
                session: self.session,
            },
        )
    }
}

impl<'de> DeserializeSeed<'de> for DTypeSerde<'_, Vec<DType>> {
    type Value = Vec<DType>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DTypeVecVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for DTypeVecVisitor<'_> {
            type Value = Vec<DType>;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("a sequence of DTypes")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut dtypes = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(dtype) =
                    seq.next_element_seed(DTypeSerde::<DType>::new(self.session))?
                {
                    dtypes.push(dtype);
                }
                Ok(dtypes)
            }
        }

        deserializer.deserialize_seq(DTypeVecVisitor {
            session: self.session,
        })
    }
}

// ============================================================================
// ExtDTypeRef Serialization
// ============================================================================

impl Serialize for ExtDTypeRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ExtDType", 3)?;
        state.serialize_field("id", self.id().as_ref())?;
        state.serialize_field("storage_dtype", self.storage_dtype())?;
        state.serialize_field(
            "metadata",
            &self
                .serialize_metadata()
                .map_err(|e| serde::ser::Error::custom(e.to_string()))?,
        )?;
        state.end()
    }
}

// ============================================================================
// ExtDTypeRef Deserialization (with session context)
// ============================================================================

impl<'de> DeserializeSeed<'de> for DTypeSerde<'_, ExtDTypeRef> {
    type Value = ExtDTypeRef;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["id", "storage_dtype", "metadata"];

        struct ExtDTypeVisitor<'a> {
            session: &'a VortexSession,
        }

        impl<'de> Visitor<'de> for ExtDTypeVisitor<'_> {
            type Value = ExtDTypeRef;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("struct ExtDType")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ExtDTypeRef, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id: Option<Arc<str>> = None;
                let mut storage_dtype: Option<DType> = None;
                let mut metadata: Option<Vec<u8>> = None;

                while let Some(key) = map.next_key::<Cow<'_, str>>()? {
                    match key.as_ref() {
                        "id" => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        "storage_dtype" => {
                            if storage_dtype.is_some() {
                                return Err(de::Error::duplicate_field("storage_dtype"));
                            }
                            storage_dtype =
                                Some(map.next_value_seed(DTypeSerde::<DType>::new(self.session))?);
                        }
                        "metadata" => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }

                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let id = ExtId::new(&id);
                let storage_dtype =
                    storage_dtype.ok_or_else(|| de::Error::missing_field("storage_dtype"))?;
                let metadata = metadata.ok_or_else(|| de::Error::missing_field("metadata"))?;

                if let Some(vtable) = self.session.dtypes().registry().find(&id) {
                    vtable.deserialize(&metadata, storage_dtype).map_err(|e| {
                        de::Error::custom(format!(
                            "failed to deserialize extension dtype {}: {}",
                            id, e
                        ))
                    })
                } else if self.session.allows_unknown() {
                    ForeignExtDType::from_parts(id, metadata, storage_dtype).map_err(|e| {
                        de::Error::custom(format!(
                            "failed to deserialize unknown extension dtype: {}",
                            e
                        ))
                    })
                } else {
                    Err(de::Error::custom(format!(
                        "unknown extension dtype id: {}",
                        id
                    )))
                }
            }
        }

        deserializer.deserialize_struct(
            "ExtDType",
            FIELDS,
            ExtDTypeVisitor {
                session: self.session,
            },
        )
    }
}
