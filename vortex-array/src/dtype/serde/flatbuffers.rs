// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::hash::Hasher;
use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::UOffsetT;
use flatbuffers::UnionWIPOffset;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use itertools::Itertools;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_map::HashTable;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldDType;
use crate::dtype::FieldNames;
use crate::dtype::MapDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::UnionVariants;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ForeignExtDType;
use crate::dtype::flatbuffers as fb;
use crate::dtype::session::DTypeSessionExt;
use crate::flatbuffers::FlatBuffer;
use crate::flatbuffers::FlatBufferRoot;
use crate::flatbuffers::WriteFlatBuffer;
use crate::flatbuffers::dtype as fbd;

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
        let names: FieldNames = fb_struct
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

        if names.len() != dtypes.len() {
            vortex_bail!(
                "length mismatch between struct names ({}) and dtypes ({})",
                names.len(),
                dtypes.len()
            );
        }

        Ok(StructFields::from_fields(names, dtypes))
    }
}

impl UnionVariants {
    /// Creates a new instance from a flatbuffer-defined object and its underlying buffer.
    fn from_fb(
        fb_union: fbd::Union<'_>,
        buffer: FlatBuffer,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let names = fb_union
            .names()
            .ok_or_else(|| vortex_err!("failed to parse union names from flatbuffer"))?
            .iter()
            .collect();

        let dtypes = fb_union
            .dtypes()
            .ok_or_else(|| vortex_err!("failed to parse union dtypes from flatbuffer"))?
            .iter()
            .map(|dt| {
                FieldDType::from(ViewedDType::from_fb_loc(
                    dt._tab.loc(),
                    buffer.clone(),
                    session.clone(),
                ))
            })
            .collect::<Vec<_>>();

        let type_ids: Vec<u8> = fb_union
            .type_ids()
            .ok_or_else(|| vortex_err!("failed to parse union type_ids from flatbuffer"))?
            .iter()
            .map(i8::cast_unsigned)
            .collect();

        UnionVariants::try_from_fields(names, dtypes, type_ids)
    }
}

impl MapDType {
    /// Creates a map dtype from a flatbuffer-defined object and its underlying buffer.
    fn from_fb(
        fb_map: fbd::Map<'_>,
        buffer: FlatBuffer,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let key = fb_map
            .key_type()
            .ok_or_else(|| vortex_err!("failed to parse map key type from flatbuffer"))?;
        let value = fb_map
            .value_type()
            .ok_or_else(|| vortex_err!("failed to parse map value type from flatbuffer"))?;

        MapDType::try_from_fields(
            FieldDType::from(ViewedDType::from_fb_loc(
                key._tab.loc(),
                buffer.clone(),
                session.clone(),
            )),
            FieldDType::from(ViewedDType::from_fb_loc(value._tab.loc(), buffer, session)),
            fb_map.keys_sorted(),
        )
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
            fb::Type::Utf8 => Ok(Self::Utf8(
                fb.type__as_utf_8()
                    .ok_or_else(|| vortex_err!("failed to parse utf-8 from flatbuffer"))?
                    .nullable()
                    .into(),
            )),
            fb::Type::Binary => Ok(Self::Binary(
                fb.type__as_binary()
                    .ok_or_else(|| vortex_err!("failed to parse binary from flatbuffer"))?
                    .nullable()
                    .into(),
            )),
            fb::Type::FixedSizeBinary => {
                let fb_fsb = fb.type__as_fixed_size_binary().ok_or_else(|| {
                    vortex_err!("failed to parse fixed-size binary from flatbuffer")
                })?;
                Ok(Self::FixedSizeBinary(
                    fb_fsb.size(),
                    fb_fsb.nullable().into(),
                ))
            }
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
            fb::Type::Map => {
                let fb_map = fb
                    .type__as_map()
                    .ok_or_else(|| vortex_err!("failed to parse map from flatbuffer"))?;
                let map = MapDType::from_fb(fb_map, vfdt.buffer().clone(), vfdt.session.clone())?;
                Ok(Self::Map(map, fb_map.nullable().into()))
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
                let variants =
                    UnionVariants::from_fb(fb_union, vfdt.buffer().clone(), vfdt.session.clone())?;
                Ok(Self::Union(variants, fb_union.nullable().into()))
            }
            fb::Type::Variant => {
                let fb_variant = fb
                    .type__as_variant()
                    .ok_or_else(|| vortex_err!("failed to parse variant from flatbuffer"))?;
                Ok(Self::Variant(fb_variant.nullable().into()))
            }
            fb::Type::Extension => {
                let fb_ext = fb
                    .type__as_extension()
                    .ok_or_else(|| vortex_err!("failed to parse extension from flatbuffer"))?;
                #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
                let id =
                    ExtId::new(fb_ext.id().ok_or_else(|| {
                        vortex_err!("failed to parse extension id from flatbuffer")
                    })?);
                let storage_dtype = fb_ext.storage_dtype().ok_or_else(|| {
                    vortex_err!("storage_dtype must be present on DType fbs message")
                })?;
                let storage_view = ViewedDType::from_fb_loc(
                    storage_dtype._tab.loc(),
                    vfdt.buffer().clone(),
                    vfdt.session.clone(),
                );
                let storage_dtype = DType::try_from(storage_view)
                    .map_err(|e| e.with_context("failed to create DType from fbs message"))?;

                let metadata = fb_ext
                    .metadata()
                    .ok_or_else(|| {
                        vortex_err!("failed to parse extension metadata from flatbuffer")
                    })?
                    .bytes();
                let ext_dtype = if let Some(vtable) = vfdt.session.dtypes().registry().get(&id) {
                    vtable.deserialize(metadata, storage_dtype)?
                } else if vfdt.session.allows_unknown() {
                    ForeignExtDType::from_parts(id, metadata.to_vec(), storage_dtype)?
                } else {
                    return Err(vortex_err!("No such DType extension ID: {}", id));
                };

                Ok(Self::Extension(ext_dtype))
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
        Interner::default().write_dtype(fbb, self)
    }
}

/// A `DType` table, identified by the offsets of its already interned children.
#[derive(PartialEq, Eq, Hash)]
enum DTypeKey {
    Null,
    Bool(Nullability),
    Primitive(PType, Nullability),
    Decimal(DecimalDType, Nullability),
    Utf8(Nullability),
    Binary(Nullability),
    FixedSizeBinary(u32, Nullability),
    List(UOffsetT, Nullability),
    FixedSizeList(UOffsetT, u32, Nullability),
    Map(UOffsetT, UOffsetT, bool, Nullability),
    Struct(UOffsetT, UOffsetT, Nullability),
    Union(UOffsetT, UOffsetT, UOffsetT, Nullability),
    Variant(Nullability),
    Extension(UOffsetT, UOffsetT, UOffsetT),
}

/// Writes every distinct `DType` table, string and vector once per builder.
///
/// Children are interned before their parent, so equal subtrees have equal offsets and each key
/// hashes in time proportional to its own fan-out.
///
/// Strings and offset vectors are keyed only by their offset and compared against the bytes
/// already in the builder, so the tables hold 4 bytes per entry instead of an owned copy.
/// A hash collision costs one extra comparison: entries are reused only on byte equality, so the
/// output never depends on hash values or the random seed.
#[derive(Default)]
struct Interner {
    dtypes: HashMap<DTypeKey, UOffsetT>,
    hasher: DefaultHashBuilder,
    strings: HashTable<UOffsetT>,
    offset_vectors: HashTable<UOffsetT>,
    byte_vectors: HashMap<Vec<u8>, UOffsetT>,
}

impl Interner {
    fn write_dtype<'fb>(
        &mut self,
        fbb: &mut FlatBufferBuilder<'fb>,
        dtype: &DType,
    ) -> VortexResult<WIPOffset<fb::DType<'fb>>> {
        let key = match dtype {
            DType::Null => DTypeKey::Null,
            DType::Bool(n) => DTypeKey::Bool(*n),
            DType::Primitive(ptype, n) => DTypeKey::Primitive(*ptype, *n),
            DType::Decimal(dt, n) => DTypeKey::Decimal(*dt, *n),
            DType::Utf8(n) => DTypeKey::Utf8(*n),
            DType::Binary(n) => DTypeKey::Binary(*n),
            DType::FixedSizeBinary(size, n) => DTypeKey::FixedSizeBinary(*size, *n),
            DType::Variant(n) => DTypeKey::Variant(*n),
            DType::List(edt, n) => DTypeKey::List(self.write_dtype(fbb, edt)?.value(), *n),
            DType::FixedSizeList(edt, size, n) => {
                DTypeKey::FixedSizeList(self.write_dtype(fbb, edt)?.value(), *size, *n)
            }
            DType::Map(map, n) => DTypeKey::Map(
                self.write_dtype(fbb, &map.key_dtype())?.value(),
                self.write_dtype(fbb, &map.value_dtype())?.value(),
                map.keys_sorted(),
                *n,
            ),
            DType::Struct(st, n) => DTypeKey::Struct(
                self.write_names(fbb, st.names()),
                self.write_dtypes(fbb, st.fields())?,
                *n,
            ),
            DType::Union(uv, n) => {
                // The FlatBuffers schema retains its original signed byte wire type for backwards
                // compatibility. Reinterpreting the bits preserves the full u8 tag range.
                DTypeKey::Union(
                    self.write_names(fbb, uv.names()),
                    self.write_dtypes(fbb, uv.variants())?,
                    self.write_bytes(fbb, uv.type_ids()),
                    *n,
                )
            }
            DType::Extension(ext) => DTypeKey::Extension(
                self.write_string(fbb, ext.id().as_ref()),
                self.write_dtype(fbb, ext.storage_dtype())?.value(),
                self.write_bytes(fbb, &ext.serialize_metadata()?),
            ),
        };

        if let Some(&offset) = self.dtypes.get(&key) {
            return Ok(WIPOffset::new(offset));
        }
        let offset = write_dtype_table(fbb, &key);
        self.dtypes.insert(key, offset.value());
        Ok(offset)
    }

    fn write_dtypes<'fb>(
        &mut self,
        fbb: &mut FlatBufferBuilder<'fb>,
        dtypes: impl Iterator<Item = DType>,
    ) -> VortexResult<UOffsetT> {
        let offsets = dtypes
            .map(|dtype| Ok(self.write_dtype(fbb, &dtype)?.value()))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(self.write_offsets(fbb, offsets))
    }

    fn write_names(&mut self, fbb: &mut FlatBufferBuilder<'_>, names: &FieldNames) -> UOffsetT {
        let offsets = names
            .iter()
            .map(|name| self.write_string(fbb, name.as_ref()))
            .collect_vec();
        self.write_offsets(fbb, offsets)
    }

    fn write_string(&mut self, fbb: &mut FlatBufferBuilder<'_>, s: &str) -> UOffsetT {
        let hash = self.hasher.hash_one(s.as_bytes());
        let written = fbb.unfinished_data();
        if let Some(&offset) = self
            .strings
            .find(hash, |&o| written_string(written, o) == s.as_bytes())
        {
            return offset;
        }
        let offset = fbb.create_string(s).value();
        let written = fbb.unfinished_data();
        let hasher = &self.hasher;
        self.strings.insert_unique(hash, offset, |&o| {
            hasher.hash_one(written_string(written, o))
        });
        offset
    }

    fn write_offsets(
        &mut self,
        fbb: &mut FlatBufferBuilder<'_>,
        offsets: Vec<UOffsetT>,
    ) -> UOffsetT {
        let hash = hash_offsets(&self.hasher, offsets.iter().copied());
        let written = fbb.unfinished_data();
        if let Some(&offset) = self.offset_vectors.find(hash, |&o| {
            written_offsets(written, o).eq(offsets.iter().copied())
        }) {
            return offset;
        }
        // The wire format of an offset vector does not depend on the type it points to.
        let typed = offsets
            .iter()
            .map(|&o| WIPOffset::<UnionWIPOffset>::new(o))
            .collect_vec();
        let offset = fbb.create_vector(&typed).value();
        let written = fbb.unfinished_data();
        let hasher = &self.hasher;
        self.offset_vectors.insert_unique(hash, offset, |&o| {
            hash_offsets(hasher, written_offsets(written, o))
        });
        offset
    }

    fn write_bytes(&mut self, fbb: &mut FlatBufferBuilder<'_>, bytes: &[u8]) -> UOffsetT {
        if let Some(&offset) = self.byte_vectors.get(bytes) {
            return offset;
        }
        let offset = fbb.create_vector(bytes).value();
        self.byte_vectors.insert(bytes.to_vec(), offset);
        offset
    }
}

fn hash_offsets(hasher: &DefaultHashBuilder, offsets: impl Iterator<Item = UOffsetT>) -> u64 {
    let mut state = hasher.build_hasher();
    let mut len = 0usize;
    for offset in offsets {
        state.write_u32(offset);
        len += 1;
    }
    state.write_usize(len);
    state.finish()
}

// The helpers below read what the builder already wrote. They rely on:
// - the FlatBuffers wire format: strings and vectors are a little-endian u32 length followed by
//   the data, and vector elements are u32 offsets relative to their own position;
// - `WIPOffset::value` counting back from the end of the in-progress buffer, as documented;
// - `unfinished_data` ending at the end of that buffer (public API, but an implementation detail).
fn read_uoffset(written: &[u8], at: usize) -> UOffsetT {
    UOffsetT::from_le_bytes([
        written[at],
        written[at + 1],
        written[at + 2],
        written[at + 3],
    ])
}

/// `offset` counts back from the end of the builder, as `WIPOffset::value` does.
fn written_string(written: &[u8], offset: UOffsetT) -> &[u8] {
    let at = written.len() - offset as usize;
    let len = read_uoffset(written, at) as usize;
    &written[at + 4..at + 4 + len]
}

/// Yields the targets of a written offset vector as `WIPOffset::value`s.
fn written_offsets(written: &[u8], offset: UOffsetT) -> impl Iterator<Item = UOffsetT> + '_ {
    let at = written.len() - offset as usize;
    (0..read_uoffset(written, at)).map(move |i| {
        let element = 4 + 4 * i;
        offset - element - read_uoffset(written, at + element as usize)
    })
}

fn write_dtype_table<'fb>(
    fbb: &mut FlatBufferBuilder<'fb>,
    key: &DTypeKey,
) -> WIPOffset<fb::DType<'fb>> {
    let (type_type, type_) = match *key {
        DTypeKey::Null => (
            fb::Type::Null,
            fb::Null::create(fbb, &fb::NullArgs {}).as_union_value(),
        ),
        DTypeKey::Bool(n) => (
            fb::Type::Bool,
            fb::Bool::create(fbb, &fb::BoolArgs { nullable: n.into() }).as_union_value(),
        ),
        DTypeKey::Primitive(ptype, n) => (
            fb::Type::Primitive,
            fb::Primitive::create(
                fbb,
                &fb::PrimitiveArgs {
                    ptype: ptype.into(),
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Decimal(dt, n) => (
            fb::Type::Decimal,
            fb::Decimal::create(
                fbb,
                &fb::DecimalArgs {
                    precision: dt.precision(),
                    scale: dt.scale(),
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Utf8(n) => (
            fb::Type::Utf8,
            fb::Utf8::create(fbb, &fb::Utf8Args { nullable: n.into() }).as_union_value(),
        ),
        DTypeKey::Binary(n) => (
            fb::Type::Binary,
            fb::Binary::create(fbb, &fb::BinaryArgs { nullable: n.into() }).as_union_value(),
        ),
        DTypeKey::FixedSizeBinary(size, n) => (
            fb::Type::FixedSizeBinary,
            fb::FixedSizeBinary::create(
                fbb,
                &fb::FixedSizeBinaryArgs {
                    size,
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Variant(n) => (
            fb::Type::Variant,
            fb::Variant::create(fbb, &fb::VariantArgs { nullable: n.into() }).as_union_value(),
        ),
        DTypeKey::List(element_type, n) => (
            fb::Type::List,
            fb::List::create(
                fbb,
                &fb::ListArgs {
                    element_type: Some(WIPOffset::new(element_type)),
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::FixedSizeList(element_type, size, n) => (
            fb::Type::FixedSizeList,
            fb::FixedSizeList::create(
                fbb,
                &fb::FixedSizeListArgs {
                    element_type: Some(WIPOffset::new(element_type)),
                    size,
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Map(key_type, value_type, keys_sorted, n) => (
            fb::Type::Map,
            fb::Map::create(
                fbb,
                &fb::MapArgs {
                    key_type: Some(WIPOffset::new(key_type)),
                    value_type: Some(WIPOffset::new(value_type)),
                    keys_sorted,
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Struct(names, dtypes, n) => (
            fb::Type::Struct_,
            fb::Struct_::create(
                fbb,
                &fb::Struct_Args {
                    names: Some(WIPOffset::new(names)),
                    dtypes: Some(WIPOffset::new(dtypes)),
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Union(names, dtypes, type_ids, n) => (
            fb::Type::Union,
            fb::Union::create(
                fbb,
                &fb::UnionArgs {
                    names: Some(WIPOffset::new(names)),
                    dtypes: Some(WIPOffset::new(dtypes)),
                    type_ids: Some(WIPOffset::new(type_ids)),
                    nullable: n.into(),
                },
            )
            .as_union_value(),
        ),
        DTypeKey::Extension(id, storage_dtype, metadata) => (
            fb::Type::Extension,
            fb::Extension::create(
                fbb,
                &fb::ExtensionArgs {
                    id: Some(WIPOffset::new(id)),
                    storage_dtype: Some(WIPOffset::new(storage_dtype)),
                    metadata: Some(WIPOffset::new(metadata)),
                },
            )
            .as_union_value(),
        ),
    };
    fb::DType::create(
        fbb,
        &fb::DTypeArgs {
            type_type,
            type_: Some(type_),
        },
    )
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
            _ => vortex_bail!(
                "flatbuffer Primitive ptype must be a known PType, got {}",
                value.0
            ),
        })
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use flatbuffers::FlatBufferBuilder;
    use flatbuffers::root;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::dtype::DType;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::UnionVariants;
    use crate::dtype::flatbuffers as fb;
    use crate::dtype::nullability::Nullability;
    use crate::dtype::serde::flatbuffers::ViewedDType;
    use crate::dtype::test::SESSION;
    use crate::flatbuffers::FlatBuffer;
    use crate::flatbuffers::WriteFlatBuffer;
    use crate::flatbuffers::WriteFlatBufferExt;

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
        roundtrip_dtype(DType::FixedSizeBinary(16, Nullability::NonNullable));
        roundtrip_dtype(DType::FixedSizeBinary(0, Nullability::Nullable));
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
        let inner_map = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::NonNullable,
        )
        .unwrap();
        let map = DType::map(
            inner_map,
            DType::Utf8(Nullability::Nullable),
            false,
            Nullability::Nullable,
        )
        .unwrap();
        roundtrip_dtype(DType::struct_([("map", map)], Nullability::NonNullable));
        roundtrip_dtype(DType::Variant(Nullability::Nullable));
    }

    #[test]
    fn equal_subtrees_are_written_once() -> VortexResult<()> {
        let leaf = DType::Primitive(PType::I64, Nullability::NonNullable);
        let inner: DType = DType::Struct(
            (0..100).map(|i| (format!("f{i}"), leaf.clone())).collect(),
            Nullability::NonNullable,
        );
        let outer = DType::Struct(
            (0..1000)
                .map(|i| (format!("c{i}"), inner.clone()))
                .collect(),
            Nullability::Nullable,
        );

        let inner_len = inner.write_flatbuffer_bytes()?.len();
        let bytes = outer.write_flatbuffer_bytes()?;
        // Each repeated field costs only its name and offsets, not another copy of `inner`.
        assert!(
            bytes.len() < inner_len + 1000 * 32,
            "outer {} bytes, inner {inner_len} bytes",
            bytes.len()
        );

        let fb_struct = root::<fb::DType>(&bytes)?
            .type__as_struct_()
            .ok_or_else(|| vortex_err!("expected struct"))?;
        let locs = fb_struct
            .dtypes()
            .ok_or_else(|| vortex_err!("expected dtypes"))?
            .iter()
            .map(|dt| dt._tab.loc())
            .collect::<HashSet<_>>();
        assert_eq!(locs.len(), 1);

        assert_eq!(
            DType::from_flatbuffer(FlatBuffer::from(bytes), &SESSION)?,
            outer
        );
        Ok(())
    }

    #[test]
    fn equal_subtrees_are_shared_across_depths() -> VortexResult<()> {
        let common = DType::struct_(
            [
                ("x", DType::Utf8(Nullability::Nullable)),
                ("y", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ],
            Nullability::Nullable,
        );
        let depth = 20;
        let mut dtype = DType::struct_([("b", common.clone())], Nullability::NonNullable);
        for _ in 0..depth {
            dtype = DType::struct_(
                [("a", dtype), ("b", common.clone())],
                Nullability::NonNullable,
            );
        }

        let bytes = dtype.write_flatbuffer_bytes()?;
        let mut node = root::<fb::DType>(&bytes)?;
        let mut common_locs = HashSet::new();
        loop {
            let fields = node
                .type__as_struct_()
                .and_then(|st| st.dtypes())
                .ok_or_else(|| vortex_err!("expected struct"))?;
            common_locs.insert(fields.get(fields.len() - 1)._tab.loc());
            if fields.len() == 1 {
                break;
            }
            node = fields.get(0);
        }
        assert_eq!(common_locs.len(), 1);

        assert_eq!(
            DType::from_flatbuffer(FlatBuffer::from(bytes), &SESSION)?,
            dtype
        );
        Ok(())
    }

    fn make_basic_union() -> DType {
        DType::Union(
            UnionVariants::new(
                ["int", "str"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            )
            .unwrap(),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn test_union_round_trip_flatbuffer() {
        roundtrip_dtype(make_basic_union());
    }

    #[test]
    fn test_union_round_trip_flatbuffer_with_nullable_variant() {
        let dtype = DType::Union(
            UnionVariants::new(
                ["null_variant", "str"].into(),
                vec![DType::Null, DType::Utf8(Nullability::NonNullable)],
            )
            .unwrap(),
            Nullability::NonNullable,
        );
        roundtrip_dtype(dtype);
    }

    #[test]
    fn test_union_round_trip_flatbuffer_with_type_id_indirection() {
        let dtype = DType::Union(
            UnionVariants::try_new(
                ["a", "b", "c"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                    DType::Bool(Nullability::NonNullable),
                ],
                vec![0, 5, u8::MAX],
            )
            .unwrap(),
            Nullability::Nullable,
        );

        let bytes = dtype.write_flatbuffer_bytes().unwrap();
        let root_fb = root::<fb::DType>(&bytes).unwrap();
        let view = ViewedDType::from_fb_loc(
            root_fb._tab.loc(),
            FlatBuffer::from(bytes.clone()),
            SESSION.clone(),
        );

        let deserialized = DType::try_from(view).unwrap();
        assert_eq!(dtype, deserialized);
        let DType::Union(uv, _) = &deserialized else {
            panic!("Expected Union");
        };
        assert_eq!(uv.type_ids(), &[0, 5, u8::MAX]);
    }

    #[test]
    fn test_nested_union_round_trip_flatbuffer() {
        let inner_union = make_basic_union();
        let struct_with_union = DType::Struct(
            StructFields::from_iter([
                ("id", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("inner", inner_union),
            ]),
            Nullability::NonNullable,
        );

        let outer_union = DType::Union(
            UnionVariants::new(
                ["plain", "nested"].into(),
                vec![DType::Utf8(Nullability::NonNullable), struct_with_union],
            )
            .unwrap(),
            Nullability::Nullable,
        );

        roundtrip_dtype(outer_union);
    }

    /// A `UnionVariants` parsed lazily from a flatbuffer must compare equal to its eager
    /// counterpart.
    #[test]
    fn test_union_viewed_dtype_equality() {
        let eager = make_basic_union();

        let bytes = eager.write_flatbuffer_bytes().unwrap();
        let root_fb = root::<fb::DType>(&bytes).unwrap();
        let view = ViewedDType::from_fb_loc(
            root_fb._tab.loc(),
            FlatBuffer::from(bytes.clone()),
            SESSION.clone(),
        );

        let viewed = DType::try_from(view).unwrap();
        assert_eq!(eager, viewed);
        assert_eq!(viewed, eager);
    }

    #[test]
    fn test_struct_malformed_flatbuffer() {
        let mut fbb = FlatBufferBuilder::new();
        let names = fbb.create_vector::<flatbuffers::WIPOffset<&str>>(&[]);
        let dtype_offsets = (0..3)
            .map(|_| {
                DType::Primitive(PType::I32, Nullability::NonNullable)
                    .write_flatbuffer(&mut fbb)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let dtypes = fbb.create_vector(&dtype_offsets);

        let struct_table = fb::Struct_::create(
            &mut fbb,
            &fb::Struct_Args {
                names: Some(names),
                dtypes: Some(dtypes),
                nullable: false,
            },
        );

        let dtype = fb::DType::create(
            &mut fbb,
            &fb::DTypeArgs {
                type_type: fb::Type::Struct_,
                type_: Some(struct_table.as_union_value()),
            },
        );
        fbb.finish_minimal(dtype);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        let buffer = FlatBuffer::align_from(ByteBuffer::from(vec).slice(start..end));

        let root_fb = root::<fb::DType>(&buffer).unwrap();
        let view = ViewedDType::from_fb_loc(root_fb._tab.loc(), buffer, SESSION.clone());

        let result = DType::try_from(view);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length mismatch"));
    }

    /// A malformed flatbuffer (here, `dtypes.len() != type_ids.len()`) must round-trip
    /// to `Err`, not panic.
    #[test]
    fn test_union_malformed_flatbuffer_errors() {
        use flatbuffers::FlatBufferBuilder;
        use vortex_buffer::ByteBuffer;

        use crate::flatbuffers::WriteFlatBuffer;

        let mut fbb = FlatBufferBuilder::new();

        // Build a Union with 2 names + 2 dtypes but only 1 type_id.
        let name_a = fbb.create_string("a");
        let name_b = fbb.create_string("b");
        let names = fbb.create_vector(&[name_a, name_b]);

        let dtype_a = DType::Primitive(PType::I32, Nullability::NonNullable)
            .write_flatbuffer(&mut fbb)
            .unwrap();
        let dtype_b = DType::Utf8(Nullability::NonNullable)
            .write_flatbuffer(&mut fbb)
            .unwrap();
        let dtypes = fbb.create_vector(&[dtype_a, dtype_b]);

        let type_ids = fbb.create_vector::<i8>(&[0]);

        let union_table = fb::Union::create(
            &mut fbb,
            &fb::UnionArgs {
                names: Some(names),
                dtypes: Some(dtypes),
                type_ids: Some(type_ids),
                nullable: false,
            },
        );

        let dtype = fb::DType::create(
            &mut fbb,
            &fb::DTypeArgs {
                type_type: fb::Type::Union,
                type_: Some(union_table.as_union_value()),
            },
        );
        fbb.finish_minimal(dtype);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        let buffer = FlatBuffer::align_from(ByteBuffer::from(vec).slice(start..end));

        let root_fb = root::<fb::DType>(&buffer).unwrap();
        let view = ViewedDType::from_fb_loc(root_fb._tab.loc(), buffer, SESSION.clone());

        let result = DType::try_from(view);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length mismatch"));
    }

    #[test]
    fn test_unknown_ptype_from_flatbuffer() {
        let result = PType::try_from(fb::PType(11));
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("flatbuffer Primitive ptype must be a known PType, got 11")
        }));
    }
}
