// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferBuilder;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WIPOffset;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::dtype as fbd;
use vortex_flatbuffers::root;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldDType;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtId;
use crate::dtype::flatbuffers as fb;
use crate::dtype::session::DTypeSessionExt;

/// A lazily evaluated DType, parsed on access from an underlying flatbuffer.
#[derive(Debug, Clone)]
pub(crate) struct ViewedDType {
    dtype: DType,
}

impl ViewedDType {
    fn from_fb(dtype: fbd::DTypeRef<'_>, session: &VortexSession) -> VortexResult<Self> {
        Ok(Self {
            dtype: DType::from_fb(dtype, session)?,
        })
    }
}

impl StructFields {
    /// Creates a new instance from a flatbuffer-defined object.
    fn from_fb(fb_struct: fbd::StructRef<'_>, session: &VortexSession) -> VortexResult<Self> {
        let names = fb_struct
            .names()?
            .ok_or_else(|| vortex_err!("failed to parse struct names from flatbuffer"))?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let dtypes = fb_struct
            .dtypes()?
            .ok_or_else(|| vortex_err!("failed to parse struct dtypes from flatbuffer"))?
            .into_iter()
            .map(|dt| Ok(FieldDType::from(ViewedDType::from_fb(dt?, session)?)))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(StructFields::from_fields(names.into(), dtypes))
    }
}

impl DType {
    fn as_fb_dtype(&self) -> VortexResult<fb::DType> {
        let dtype_union = match self {
            Self::Null => fb::Type::Null(Box::new(fb::Null {})),
            Self::Bool(n) => fb::Type::Bool(Box::new(fb::Bool {
                nullable: (*n).into(),
            })),
            Self::Primitive(ptype, n) => fb::Type::Primitive(Box::new(fb::Primitive {
                ptype: (*ptype).into(),
                nullable: (*n).into(),
            })),
            Self::Decimal(dt, n) => fb::Type::Decimal(Box::new(fb::Decimal {
                precision: dt.precision(),
                scale: dt.scale(),
                nullable: (*n).into(),
            })),
            Self::Utf8(n) => fb::Type::Utf8(Box::new(fb::Utf8 {
                nullable: (*n).into(),
            })),
            Self::Binary(n) => fb::Type::Binary(Box::new(fb::Binary {
                nullable: (*n).into(),
            })),
            Self::Struct(st, n) => fb::Type::Struct(Box::new(fb::Struct {
                names: Some(st.names().iter().map(|name| name.to_string()).collect()),
                dtypes: Some(
                    st.fields()
                        .map(|dtype| dtype.as_fb_dtype())
                        .collect::<VortexResult<Vec<_>>>()?,
                ),
                nullable: (*n).into(),
            })),
            Self::List(edt, n) => fb::Type::List(Box::new(fb::List {
                element_type: Some(Box::new(edt.as_ref().as_fb_dtype()?)),
                nullable: (*n).into(),
            })),
            Self::FixedSizeList(edt, size, n) => {
                fb::Type::FixedSizeList(Box::new(fb::FixedSizeList {
                    element_type: Some(Box::new(edt.as_ref().as_fb_dtype()?)),
                    size: *size,
                    nullable: (*n).into(),
                }))
            }
            Self::Extension(ext) => fb::Type::Extension(Box::new(fb::Extension {
                id: Some(ext.id().as_ref().to_string()),
                storage_dtype: Some(Box::new(ext.storage_dtype().as_fb_dtype()?)),
                metadata: Some(ext.serialize_metadata()?),
            })),
            Self::Variant(n) => fb::Type::Variant(Box::new(fb::Variant {
                nullable: (*n).into(),
            })),
        };

        Ok(fb::DType {
            type_: Some(dtype_union),
        })
    }

    fn from_fb(fb_dtype: fbd::DTypeRef<'_>, session: &VortexSession) -> VortexResult<Self> {
        let Some(dtype) = fb_dtype.type_()? else {
            return Err(vortex_err!("failed to parse DType union from flatbuffer"));
        };

        match dtype {
            fbd::TypeRef::Null(_) => Ok(Self::Null),
            fbd::TypeRef::Bool(fb_bool) => Ok(Self::Bool(fb_bool.nullable()?.into())),
            fbd::TypeRef::Primitive(fb_primitive) => Ok(Self::Primitive(
                fb_primitive.ptype()?.try_into()?,
                fb_primitive.nullable()?.into(),
            )),
            fbd::TypeRef::Decimal(fb_decimal) => Ok(Self::Decimal(
                DecimalDType::try_new(fb_decimal.precision()?, fb_decimal.scale()?)?,
                fb_decimal.nullable()?.into(),
            )),
            fbd::TypeRef::Binary(fb_binary) => Ok(Self::Binary(fb_binary.nullable()?.into())),
            fbd::TypeRef::Utf8(fb_utf8) => Ok(Self::Utf8(fb_utf8.nullable()?.into())),
            fbd::TypeRef::List(fb_list) => {
                let element_dtype = fb_list
                    .element_type()?
                    .map(|dtype| Self::from_fb(dtype, session))
                    .transpose()?
                    .ok_or_else(|| {
                        vortex_err!("failed to parse list element type from flatbuffer")
                    })?;

                Ok(Self::List(
                    Arc::new(element_dtype),
                    fb_list.nullable()?.into(),
                ))
            }
            fbd::TypeRef::FixedSizeList(fb_fixed_size_list) => {
                let element_dtype = fb_fixed_size_list
                    .element_type()?
                    .map(|dtype| Self::from_fb(dtype, session))
                    .transpose()?
                    .ok_or_else(|| {
                        vortex_err!("failed to parse list element type from flatbuffer")
                    })?;

                Ok(Self::FixedSizeList(
                    Arc::new(element_dtype),
                    fb_fixed_size_list.size()?,
                    fb_fixed_size_list.nullable()?.into(),
                ))
            }
            fbd::TypeRef::Struct(fb_struct) => {
                let nullable = fb_struct.nullable()?;
                Ok(Self::Struct(
                    StructFields::from_fb(fb_struct, session)?,
                    nullable.into(),
                ))
            }
            fbd::TypeRef::Extension(fb_ext) => {
                let id = ExtId::new_arc(
                    fb_ext
                        .id()?
                        .ok_or_else(|| vortex_err!("failed to parse extension id from flatbuffer"))?
                        .into(),
                );
                let storage_dtype = fb_ext
                    .storage_dtype()?
                    .map(|dtype| Self::from_fb(dtype, session))
                    .transpose()?
                    .ok_or_else(
                        || vortex_err!(Serde: "storage_dtype must be present on DType fbs message"),
                    )?;

                let vtable = session
                    .dtypes()
                    .registry()
                    .find(&id)
                    .ok_or_else(|| vortex_err!("No such DType extension ID: {}", id))?;
                let ext_dtype = vtable.deserialize(
                    fb_ext.metadata()?.ok_or_else(|| {
                        vortex_err!("failed to parse extension metadata from flatbuffer")
                    })?,
                    storage_dtype,
                )?;

                Ok(Self::Extension(ext_dtype))
            }
            fbd::TypeRef::Variant(fb_variant) => Ok(Self::Variant(fb_variant.nullable()?.into())),
        }
    }

    /// Create a [`DType`] from a flatbuffer buffer.
    pub fn from_flatbuffer(buffer: FlatBuffer, session: &VortexSession) -> VortexResult<Self> {
        let fb_dtype = root::<fbd::DTypeRef<'_>>(buffer.as_ref())?;
        Self::from_fb(fb_dtype, session)
    }
}

impl TryFrom<ViewedDType> for DType {
    type Error = VortexError;

    fn try_from(vfdt: ViewedDType) -> Result<Self, Self::Error> {
        Ok(vfdt.dtype)
    }
}

impl FlatBufferRoot for DType {}

impl WriteFlatBuffer for DType {
    type Target = fb::DType;

    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>> {
        let dtype = self.as_fb_dtype()?;
        Ok(fb::DType::create(fbb, dtype.type_))
    }
}

impl From<PType> for fb::PType {
    fn from(value: PType) -> Self {
        match value {
            PType::U8 => Self::U8,
            PType::U16 => Self::U16,
            PType::U32 => Self::U32,
            PType::U64 => Self::U64,
            PType::I8 => Self::I8,
            PType::I16 => Self::I16,
            PType::I32 => Self::I32,
            PType::I64 => Self::I64,
            PType::F16 => Self::F16,
            PType::F32 => Self::F32,
            PType::F64 => Self::F64,
        }
    }
}

impl TryFrom<fb::PType> for PType {
    type Error = VortexError;

    fn try_from(value: fb::PType) -> Result<Self, Self::Error> {
        Ok(match value {
            fb::PType::U8 => Self::U8,
            fb::PType::U16 => Self::U16,
            fb::PType::U32 => Self::U32,
            fb::PType::U64 => Self::U64,
            fb::PType::I8 => Self::I8,
            fb::PType::I16 => Self::I16,
            fb::PType::I32 => Self::I32,
            fb::PType::I64 => Self::I64,
            fb::PType::F16 => Self::F16,
            fb::PType::F32 => Self::F32,
            fb::PType::F64 => Self::F64,
        })
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_flatbuffers::WriteFlatBufferExt;

    use crate::dtype::DType;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::nullability::Nullability;
    use crate::dtype::test::SESSION;

    fn roundtrip_dtype(dtype: DType) {
        let bytes = dtype.write_flatbuffer_bytes().unwrap();
        let deserialized = DType::from_flatbuffer(bytes, &SESSION).unwrap();
        assert_eq!(dtype, deserialized);
    }

    #[test]
    fn roundtrip() {
        roundtrip_dtype(DType::Null);
        roundtrip_dtype(DType::Bool(Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::U8, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::U16, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::U32, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::U64, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::I8, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::I16, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::I32, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::I64, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::F16, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::F32, Nullability::NonNullable));
        roundtrip_dtype(DType::Primitive(PType::F64, Nullability::NonNullable));
        roundtrip_dtype(DType::Binary(Nullability::NonNullable));
        roundtrip_dtype(DType::Utf8(Nullability::NonNullable));
        roundtrip_dtype(DType::List(
            Arc::new(DType::Primitive(PType::F32, Nullability::Nullable)),
            Nullability::NonNullable,
        ));
        roundtrip_dtype(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::Nullable)),
            2,
            Nullability::NonNullable,
        ));
        roundtrip_dtype(DType::Struct(
            StructFields::new(
                ["strings", "ints"].into(),
                vec![
                    DType::Utf8(Nullability::NonNullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        ));
        roundtrip_dtype(DType::Variant(Nullability::Nullable));
    }
}
