// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use itertools::Itertools;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::dtype as fbd;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldDType;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ForeignExtDType;
use crate::dtype::flatbuffers as fb;
use crate::dtype::session::DTypeSessionExt;

/// A lazily evaluated DType, parsed on access from an underlying flatbuffer.
#[derive(Debug, Clone)]
pub(crate) struct ViewedDType {
    /// Underlying flatbuffer
    flatbuffer: FlatBuffer,
    /// Location of the dtype data inside the underlying buffer
    flatbuffer_loc: usize,
    /// The Vortex session used to resolve extensions
    session: VortexSession,
}

impl ViewedDType {
    /// Create a [`ViewedDType`] from a buffer and a flatbuffer location
    fn from_fb_loc(location: usize, buffer: FlatBuffer, session: VortexSession) -> Self {
        Self {
            flatbuffer: buffer,
            flatbuffer_loc: location,
            session,
        }
    }

    /// The viewed [`fbd::DType`] instance.
    fn flatbuffer(&self) -> fbd::DType<'_> {
        unsafe { fbd::DType::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    /// Returns the underlying shared buffer
    fn buffer(&self) -> &FlatBuffer {
        &self.flatbuffer
    }
}

impl StructFields {
    /// Creates a new instance from a flatbuffer-defined object and its underlying buffer.
    fn from_fb(
        fb_struct: fbd::Struct_<'_>,
        buffer: FlatBuffer,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let names = fb_struct
            .names()
            .ok_or_else(|| vortex_err!("failed to parse struct names from flatbuffer"))?
            .iter()
            .collect();

        let dtypes = fb_struct
            .dtypes()
            .ok_or_else(|| vortex_err!("failed to parse struct dtypes from flatbuffer"))?
            .iter()
            .map(|dt| {
                FieldDType::from(ViewedDType::from_fb_loc(
                    dt._tab.loc(),
                    buffer.clone(),
                    session.clone(),
                ))
            })
            .collect::<Vec<_>>();

        Ok(StructFields::from_fields(names, dtypes))
    }
}

impl DType {
    /// Create a [`DType`] from a flatbuffer buffer.
    pub fn from_flatbuffer(buffer: FlatBuffer, session: &VortexSession) -> VortexResult<Self> {
        let fb_loc = {
            let fb_dtype = root::<fbd::DType>(&buffer)?;
            fb_dtype._tab.loc()
        };
        let view = ViewedDType::from_fb_loc(fb_loc, buffer, session.clone());
        Self::try_from(view)
    }
}

impl TryFrom<ViewedDType> for DType {
    type Error = VortexError;

    fn try_from(vfdt: ViewedDType) -> Result<Self, Self::Error> {
        let fb = vfdt.flatbuffer();
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
            fb::Type::Decimal => {
                let fb_decimal = fb
                    .type__as_decimal()
                    .ok_or_else(|| vortex_err!("failed to parse decimal dtype from flatbuffer"))?;
                Ok(Self::Decimal(
                    DecimalDType::try_new(fb_decimal.precision(), fb_decimal.scale())?,
                    fb_decimal.nullable().into(),
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

                let list_element = fb_list.element_type().ok_or_else(|| {
                    vortex_err!("failed to parse list element type from flatbuffer")
                })?;
                let element_dtype = Self::try_from(ViewedDType::from_fb_loc(
                    list_element._tab.loc(),
                    vfdt.buffer().clone(),
                    vfdt.session.clone(),
                ))?;
                Ok(Self::List(
                    Arc::new(element_dtype),
                    fb_list.nullable().into(),
                ))
            }
            fb::Type::FixedSizeList => {
                let fb_fixed_size_list = fb.type__as_fixed_size_list().ok_or_else(|| {
                    vortex_err!("failed to parse fixed-size list from flatbuffer")
                })?;

                let list_element = fb_fixed_size_list.element_type().ok_or_else(|| {
                    vortex_err!("failed to parse list element type from flatbuffer")
                })?;
                let element_dtype = Self::try_from(ViewedDType::from_fb_loc(
                    list_element._tab.loc(),
                    vfdt.buffer().clone(),
                    vfdt.session.clone(),
                ))?;
                Ok(Self::FixedSizeList(
                    Arc::new(element_dtype),
                    fb_fixed_size_list.size(),
                    fb_fixed_size_list.nullable().into(),
                ))
            }
            fb::Type::Struct_ => {
                let fb_struct = fb
                    .type__as_struct_()
                    .ok_or_else(|| vortex_err!("failed to parse struct from flatbuffer"))?;
                let struct_dtype =
                    StructFields::from_fb(fb_struct, vfdt.buffer().clone(), vfdt.session.clone())?;

                Ok(Self::Struct(struct_dtype, fb_struct.nullable().into()))
            }
            fb::Type::Union => {
                let fb_union = fb
                    .type__as_union()
                    .ok_or_else(|| vortex_err!("failed to parse union from flatbuffer"))?;
                Ok(Self::Union(fb_union.nullable().into()))
            }
            fb::Type::Extension => {
                let fb_ext = fb
                    .type__as_extension()
                    .ok_or_else(|| vortex_err!("failed to parse extension from flatbuffer"))?;
                let id =
                    ExtId::new(fb_ext.id().ok_or_else(|| {
                        vortex_err!("failed to parse extension id from flatbuffer")
                    })?);
                let storage_dtype = fb_ext.storage_dtype().ok_or_else(|| {
                    vortex_err!(
                Serde: "storage_dtype must be present on DType fbs message")
                })?;
                let storage_view = ViewedDType::from_fb_loc(
                    storage_dtype._tab.loc(),
                    vfdt.buffer().clone(),
                    vfdt.session.clone(),
                );
                let storage_dtype = DType::try_from(storage_view)
                    .map_err(|e| vortex_err!("failed to create DType from fbs message: {e}"))?;

                let metadata = fb_ext
                    .metadata()
                    .ok_or_else(|| {
                        vortex_err!("failed to parse extension metadata from flatbuffer")
                    })?
                    .bytes();
                let ext_dtype = if let Some(vtable) = vfdt.session.dtypes().registry().find(&id) {
                    vtable.deserialize(metadata, storage_dtype)?
                } else if vfdt.session.allows_unknown() {
                    ForeignExtDType::from_parts(id, metadata.to_vec(), storage_dtype)?
                } else {
                    return Err(vortex_err!("No such DType extension ID: {}", id));
                };

                Ok(Self::Extension(ext_dtype))
            }
            fb::Type::Variant => {
                let fb_variant = fb
                    .type__as_variant()
                    .ok_or_else(|| vortex_err!("failed to parse variant from flatbuffer"))?;
                Ok(Self::Variant(fb_variant.nullable().into()))
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
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
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
            Self::Decimal(dt, n) => fb::Decimal::create(
                fbb,
                &fb::DecimalArgs {
                    precision: dt.precision(),
                    scale: dt.scale(),
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
                    .fields()
                    .map(|dtype| dtype.write_flatbuffer(fbb))
                    .collect::<VortexResult<Vec<_>>>()?;
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
            Self::Union(n) => fb::Union::create(
                fbb,
                &fb::UnionArgs {
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
            Self::List(edt, n) => {
                let element_type = Some(edt.as_ref().write_flatbuffer(fbb)?);
                fb::List::create(
                    fbb,
                    &fb::ListArgs {
                        element_type,
                        nullable: (*n).into(),
                    },
                )
                .as_union_value()
            }
            Self::FixedSizeList(edt, size, n) => {
                let element_type = Some(edt.as_ref().write_flatbuffer(fbb)?);
                fb::FixedSizeList::create(
                    fbb,
                    &fb::FixedSizeListArgs {
                        element_type,
                        size: *size,
                        nullable: (*n).into(),
                    },
                )
                .as_union_value()
            }
            Self::Extension(ext) => {
                let id = Some(fbb.create_string(ext.id().as_ref()));
                let storage_dtype = Some(ext.storage_dtype().write_flatbuffer(fbb)?);
                let metadata = Some(fbb.create_vector(&ext.serialize_metadata()?));
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
            Self::Variant(n) => fb::Variant::create(
                fbb,
                &fb::VariantArgs {
                    nullable: (*n).into(),
                },
            )
            .as_union_value(),
        };

        let dtype_type = match self {
            Self::Null => fb::Type::Null,
            Self::Bool(_) => fb::Type::Bool,
            Self::Primitive(..) => fb::Type::Primitive,
            Self::Decimal(..) => fb::Type::Decimal,
            Self::Utf8(_) => fb::Type::Utf8,
            Self::Binary(_) => fb::Type::Binary,
            Self::Struct(..) => fb::Type::Struct_,
            Self::Union(..) => fb::Type::Union,
            Self::List(..) => fb::Type::List,
            Self::FixedSizeList(..) => fb::Type::FixedSizeList,
            Self::Extension { .. } => fb::Type::Extension,
            Self::Variant(_) => fb::Type::Variant,
        };

        Ok(fb::DType::create(
            fbb,
            &fb::DTypeArgs {
                type_type: dtype_type,
                type_: Some(dtype_union),
            },
        ))
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
            _ => vortex_bail!(Serde: "Unknown PType variant"),
        })
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use flatbuffers::root;
    use vortex_flatbuffers::FlatBuffer;
    use vortex_flatbuffers::WriteFlatBufferExt;

    use crate::dtype::DType;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::flatbuffers as fb;
    use crate::dtype::nullability::Nullability;
    use crate::dtype::serde::flatbuffers::ViewedDType;
    use crate::dtype::test::SESSION;

    fn roundtrip_dtype(dtype: DType) {
        let bytes = dtype.write_flatbuffer_bytes().unwrap();
        let root_fb = root::<fb::DType>(&bytes).unwrap();
        let view = ViewedDType::from_fb_loc(
            root_fb._tab.loc(),
            FlatBuffer::from(bytes.clone()),
            SESSION.clone(),
        );

        let deserialized = DType::try_from(view).unwrap();
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
