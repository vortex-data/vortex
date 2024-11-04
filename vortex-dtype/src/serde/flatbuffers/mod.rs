use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_flatbuffers::{FlatBufferRoot, WriteFlatBuffer};

use crate::{flatbuffers as fb, DType, ExtDType, ExtID, ExtMetadata, PType, StructDType};

mod project;
pub use project::*;

impl TryFrom<fb::DType<'_>> for DType {
    type Error = VortexError;

    fn try_from(fb: fb::DType<'_>) -> Result<Self, Self::Error> {
        match fb.type_type() {
            fb::Type::Null => Ok(Self::Null),
            fb::Type::Bool => Ok(Self::Bool(
                fb.type__as_bool()
                    .ok_or_else(|| vortex_err!("failed to parse bool from flatbuffer"))?
                    .nullable()
                    .into(),
            )),
            fb::Type::Primitive => {
                let fb_primitive = fb
                    .type__as_primitive()
                    .ok_or_else(|| vortex_err!("failed to parse primitive from flatbuffer"))?;
                Ok(Self::Primitive(
                    fb_primitive.ptype().try_into()?,
                    fb_primitive.nullable().into(),
                ))
            }
            fb::Type::Binary => Ok(Self::Binary(
                fb.type__as_binary()
                    .ok_or_else(|| vortex_err!("failed to parse binary from flatbuffer"))?
                    .nullable()
                    .into(),
            )),
            fb::Type::Utf8 => Ok(Self::Utf8(
                fb.type__as_utf_8()
                    .ok_or_else(|| vortex_err!("failed to parse utf-8 from flatbuffer"))?
                    .nullable()
                    .into(),
            )),
            fb::Type::List => {
                let fb_list = fb
                    .type__as_list()
                    .ok_or_else(|| vortex_err!("failed to parse list from flatbuffer"))?;
                let element_dtype = Self::try_from(fb_list.element_type().ok_or_else(|| {
                    vortex_err!("failed to parse list element type from flatbuffer")
                })?)?;
                Ok(Self::List(
                    Arc::new(element_dtype),
                    fb_list.nullable().into(),
                ))
            }
            fb::Type::Struct_ => {
                let fb_struct = fb
                    .type__as_struct_()
                    .ok_or_else(|| vortex_err!("failed to parse struct from flatbuffer"))?;
                let names = fb_struct
                    .names()
                    .ok_or_else(|| vortex_err!("failed to parse struct names from flatbuffer"))?
                    .iter()
                    .map(|n| (*n).into())
                    .collect_vec()
                    .into();
                let dtypes: Vec<Self> = fb_struct
                    .dtypes()
                    .ok_or_else(|| vortex_err!("failed to parse struct dtypes from flatbuffer"))?
                    .iter()
                    .map(Self::try_from)
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(Self::Struct(
                    StructDType::new(names, dtypes),
                    fb_struct.nullable().into(),
                ))
            }
            fb::Type::Extension => {
                let fb_ext = fb
                    .type__as_extension()
                    .ok_or_else(|| vortex_err!("failed to parse extension from flatbuffer"))?;
                let id =
                    ExtID::from(fb_ext.id().ok_or_else(|| {
                        vortex_err!("failed to parse extension id from flatbuffer")
                    })?);
                let metadata = fb_ext.metadata().map(|m| ExtMetadata::from(m.bytes()));
                Ok(Self::Extension(Arc::new(ExtDType::new(
                    id,
                    Arc::new(
                        DType::try_from(fb_ext.storage_dtype().ok_or_else(|| {
                            vortex_err!(
                        InvalidSerde: "storage_dtype must be present on DType fbs message")
                        })?)
                        .map_err(|e| vortex_err!("failed to create DType from fbs message: {e}"))?,
                    ),
                    metadata,
                ))))
            }
            _ => Err(vortex_err!("Unknown DType variant")),
        }
    }
}

impl FlatBufferRoot for DType {}
impl WriteFlatBuffer for DType {
    type Target<'a> = fb::DType<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let dtype_union = match self {
            Self::Null => fb::Null::create(fbb, &fb::NullArgs {}).as_union_value(),
            Self::Bool(n) => fb::Bool::create(
                fbb,
                &fb::BoolArgs {
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
            Self::Primitive(ptype, n) => fb::Primitive::create(
                fbb,
                &fb::PrimitiveArgs {
                    ptype: (*ptype).into(),
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
            Self::Utf8(n) => fb::Utf8::create(
                fbb,
                &fb::Utf8Args {
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
            Self::Binary(n) => fb::Binary::create(
                fbb,
                &fb::BinaryArgs {
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
            Self::Struct(st, n) => {
                let names = st
                    .names()
                    .iter()
                    .map(|n| fbb.create_string(n.as_ref()))
                    .collect_vec();
                let names = Some(fbb.create_vector(&names));

                let dtypes = st
                    .dtypes()
                    .iter()
                    .map(|dtype| dtype.write_flatbuffer(fbb))
                    .collect_vec();
                let dtypes = Some(fbb.create_vector(&dtypes));

                fb::Struct_::create(
                    fbb,
                    &fb::Struct_Args {
                        names,
                        dtypes,
                        nullable: (*n).into(),
                    },
                )
                .as_union_value()
            }
            Self::List(edt, n) => {
                let element_type = Some(edt.as_ref().write_flatbuffer(fbb));
                fb::List::create(
                    fbb,
                    &fb::ListArgs {
                        element_type,
                        nullable: (*n).into(),
                    },
                )
                .as_union_value()
            }
            Self::Extension(ext) => {
                let id = Some(fbb.create_string(ext.id().as_ref()));
                let storage_dtype = Some(ext.storage_dtype().write_flatbuffer(fbb));
                let metadata = ext.metadata().map(|m| fbb.create_vector(m.as_ref()));
                fb::Extension::create(
                    fbb,
                    &fb::ExtensionArgs {
                        id,
                        storage_dtype,
                        metadata,
                    },
                )
                .as_union_value()
            }
        };

        let dtype_type = match self {
            Self::Null => fb::Type::Null,
            Self::Bool(_) => fb::Type::Bool,
            Self::Primitive(..) => fb::Type::Primitive,
            Self::Utf8(_) => fb::Type::Utf8,
            Self::Binary(_) => fb::Type::Binary,
            Self::Struct(..) => fb::Type::Struct_,
            Self::List(..) => fb::Type::List,
            Self::Extension { .. } => fb::Type::Extension,
        };

        fb::DType::create(
            fbb,
            &fb::DTypeArgs {
                type_type: dtype_type,
                type_: Some(dtype_union),
            },
        )
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
            _ => vortex_bail!(InvalidSerde: "Unknown PType variant"),
        })
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use flatbuffers::root;
    use vortex_flatbuffers::FlatBufferToBytes;

    use crate::nullability::Nullability;
    use crate::{flatbuffers as fb, DType, PType, StructDType};

    fn roundtrip_dtype(dtype: DType) {
        let bytes = dtype.with_flatbuffer_bytes(|bytes| bytes.to_vec());
        let deserialized = DType::try_from(root::<fb::DType>(&bytes).unwrap()).unwrap();
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
        roundtrip_dtype(DType::Struct(
            StructDType::new(
                ["strings".into(), "ints".into()].into(),
                vec![
                    DType::Utf8(Nullability::NonNullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        ))
    }
}
