// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
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
use serde::de::Visitor;
use serde::ser::SerializeStruct;

use crate::DType;
use crate::ExtID;
use crate::Nullability;
use crate::extension::ExtDTypeRef;
use crate::session::DTypeSession;
use crate::session::ExtDTypeRegistry;

/// Since we need to use the [`DTypeSession`] as a seed, we can't derive
/// Serialize/Deserialize for DType. Instead, we hand-roll the impls here.
struct DTypeSeed<'a, T> {
    session: &'a DTypeSession,
    _marker: PhantomData<T>,
}

impl<'a, T> DTypeSeed<'a, T> {
    fn new(session: &'a DTypeSession) -> Self {
        Self {
            session,
            _marker: Default::default(),
        }
    }
}

impl<'de> DeserializeSeed<'de> for DTypeSeed<'_, DType> {
    type Value = DType;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DTypeVisitor<'a> {
            session: &'a DTypeSession,
        }

        impl<'de, 'a> Visitor<'de> for DTypeVisitor<'a> {
            type Value = DType;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("enum DType")
            }

            fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
            where
                A: EnumAccess<'de>,
            {
                use serde::de::VariantAccess;

                let (variant, access) = data.variant::<&str>()?;
                match variant {
                    // Unit variants
                    "Null" => {
                        access.unit_variant()?;
                        Ok(DType::Null)
                    }
                    "Bool" => {
                        access.unit_variant()?;
                        Ok(DType::Bool)
                    }
                    // Newtype variants with regular Deserialize
                    "Int" => {
                        let width = access.newtype_variant()?;
                        Ok(DType::Int(width))
                    }
                    "Float" => {
                        let width = access.newtype_variant()?;
                        Ok(DType::Float(width))
                    }
                    // Newtype variant that needs the seed/context
                    "Ext" => {
                        let ext = access.newtype_variant_seed::<ExtDTypeRef>(DTypeSeed {
                            session: self.session,
                            _marker: Default::default(),
                        })?;
                        Ok(DType::Ext(ext))
                    }
                    _ => Err(de::Error::unknown_variant(
                        variant,
                        &["Null", "Bool", "Int", "Float", "Ext"],
                    )),
                }
            }
        }
    }
}

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
                .options_ref()
                .serialize()
                .map_err(|e| serde::ser::Error::custom(e.to_string()))?,
        )?;
        state.end()
    }
}

impl<'de> DeserializeSeed<'de> for DTypeSeed<'_, ExtDTypeRef> {
    type Value = ExtDTypeRef;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["id", "storage_dtype", "metadata"];

        struct ExtDTypeVisitor<'r> {
            session: &'r DTypeSession,
        }

        impl<'de> Visitor<'de> for ExtDTypeVisitor<'_> {
            type Value = ExtDTypeRef;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("struct ExtDType")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ExtDTypeRef, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id = None;
                let mut storage_dtype = None;
                let mut metadata = None;

                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "id" => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value::<Arc<str>>()?);
                        }
                        "storage_dtype" => {
                            if storage_dtype.is_some() {
                                return Err(de::Error::duplicate_field("storage_dtype"));
                            }
                            storage_dtype =
                                Some(map.next_value_seed::<DType>(DTypeSeed::new(self.session))?);
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
                let id = ExtID::new_arc(id);
                let vtable = self.session.registry().find(&id).ok_or_else(|| {
                    de::Error::custom(format!("unknown extension dtype id: {}", id))
                })?;

                let storage_dtype: DType =
                    storage_dtype.ok_or_else(|| de::Error::missing_field("storage_dtype"))?;
                let metadata: Vec<u8> =
                    metadata.ok_or_else(|| de::Error::missing_field("metadata"))?;

                vtable.deserialize(&metadata, storage_dtype).map_err(|e| {
                    de::Error::custom(format!(
                        "failed to deserialize extension dtype {}: {}",
                        id, e
                    ))
                })
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
