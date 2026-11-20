// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extracts a path of object fields from unshredded Parquet Variant storage, reading the
//! [Variant encoding] in place.
//!
//! This covers the extractions a scan runs most: a string or an integer under a few object
//! fields, and structs and lists of them. Reading the bytes in place skips what the general
//! `variant_get` kernel does for every row and every path: building an Arrow array of the
//! storage, a `Variant` over each row, and a `Variant` over each object on the way down. A struct
//! reads the fields of each object in one pass, so it extracts several paths at once.
//!
//! The values that need no conversion are read from their bytes. Every other value is decoded and
//! converted with the `Variant` method that the general kernel converts it with, so the result is
//! the kernel's whatever a row holds. [`extract`] returns `None` only for an extraction it does
//! not cover at all, or for bytes that are not what the encoding says they are, and the caller
//! then falls back to the general kernel.
//!
//! [Variant encoding]: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md

use std::sync::Arc;

use parquet_variant::Variant;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_mask::Mask;

/// Extracts `path`, a path of object fields, from each row of unshredded storage as `dtype`.
///
/// `metadata` and `value` are the storage children, and `valid` the rows that hold a value. Each
/// value converts to `dtype` as the general kernel converts it, and a missing path, or a row
/// without a value, is a null. Returns `None` if [`covers`] does not cover `dtype`, or if the
/// bytes are not what the encoding says they are.
pub(crate) fn extract(
    metadata: &VarBinViewArray,
    value: &VarBinViewArray,
    valid: &Mask,
    path: &[&str],
    dtype: &DType,
) -> Option<ArrayRef> {
    let mut names = path
        .iter()
        .copied()
        .map(FieldName::from)
        .collect::<Vec<_>>();
    let mut builder = Builder::new(dtype, &mut names, value.len())?;
    let mut cache = FieldCache::new(&names);

    let depth = path.len();
    match &mut builder {
        Builder::Utf8(utf8) => append_rows(utf8, &mut cache, metadata, value, valid, depth)?,
        Builder::Int64(int64) => append_rows(int64, &mut cache, metadata, value, valid, depth)?,
        nested => append_rows(nested, &mut cache, metadata, value, valid, depth)?,
    }

    builder.finish()
}

/// Whether [`extract`] covers extracting values as `dtype`.
///
/// It covers strings, 64-bit integers, and structs and lists of them. The fields of a struct and
/// the elements of a list must be nullable, as a value that does not convert is a null. A struct
/// without fields is left to the general kernel, which cannot build one.
pub(crate) fn covers(dtype: &DType) -> bool {
    match dtype {
        DType::Utf8(_) | DType::Primitive(PType::I64, _) => true,
        DType::Struct(fields, _) => {
            fields.nfields() > 0
                && fields
                    .fields()
                    .all(|field| field.is_nullable() && covers(&field))
        }
        DType::List(element, _) => element.is_nullable() && covers(element),
        _ => false,
    }
}

/// Follows the first `depth` names of the cached names from `value`.
///
/// Returns `Some(None)` if the path does not exist in the value, and `None` for bytes that are
/// cut short.
fn follow<'v>(fields: &Fields<'_>, mut value: &'v [u8], depth: usize) -> Option<Option<&'v [u8]>> {
    for name in 0..depth {
        // A value that is not an object has no fields.
        let Some(object) = Object::new(value)? else {
            return Some(None);
        };
        match fields.find(&object, name)? {
            Some(field) => value = field,
            None => return Some(None),
        }
    }
    Some(Some(value))
}

/// A dictionary without names. A primitive value has no fields, so it never reads the dictionary.
const EMPTY_METADATA: &[u8] = &[0x01, 0, 0];

/// Decodes the primitive `value` for a conversion of the general kernel. Returns `None` for bytes
/// that do not decode.
fn decode(value: &[u8]) -> Option<Variant<'static, '_>> {
    Variant::try_new(EMPTY_METADATA, value).ok()
}

/// Appends the value of each row, which it converts to its dtype as the general kernel converts
/// it.
///
/// The kernel converts a value to a string with [`Variant::as_string`], to an integer with
/// [`Variant::as_int64`], to a struct with [`Variant::as_object`], and to a list with
/// [`Variant::as_list`]. A value that does not convert is a null.
trait Append {
    /// Appends `value`, converted to the builder's dtype. Returns `None` for bytes that are cut
    /// short or do not decode.
    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Option<()>;

    fn append_null(&mut self);
}

/// Appends the extracted value of each row of the storage to `builder`.
///
/// Generic over the builder, so that the loop of a string or an integer target is compiled for
/// that builder alone.
fn append_rows<'a, B: Append>(
    builder: &mut B,
    cache: &mut FieldCache<'a>,
    metadata: &'a VarBinViewArray,
    value: &VarBinViewArray,
    valid: &Mask,
    depth: usize,
) -> Option<()> {
    for idx in 0..value.len() {
        if !valid.value(idx) {
            builder.append_null();
            continue;
        }
        let fields = cache.resolve(metadata, idx)?;
        match follow(&fields, bytes_at(value, idx), depth)? {
            Some(value) => builder.append(&fields, value)?,
            None => builder.append_null(),
        }
    }
    Some(())
}

/// Builds an extracted array of any dtype that [`covers`] covers.
enum Builder {
    Utf8(Utf8Builder),
    Int64(Int64Builder),
    /// Reads each field from the field of the object of the same name.
    Struct {
        names: FieldNames,
        /// The builder of each field, with the index of its name in the [`FieldCache`].
        fields: Vec<(usize, Builder)>,
        validity: BitBufferMut,
    },
    /// Reads each element from the element of the array.
    List {
        offsets: BufferMut<u64>,
        elements: Box<Builder>,
        validity: BitBufferMut,
    },
}

impl Builder {
    /// A builder of `dtype`, which adds the field names of the structs in `dtype` to `names`.
    ///
    /// Returns `None` if [`covers`] does not cover `dtype`.
    fn new(dtype: &DType, names: &mut Vec<FieldName>, capacity: usize) -> Option<Self> {
        if !covers(dtype) {
            return None;
        }
        Some(match dtype {
            DType::Utf8(_) => Self::Utf8(Utf8Builder::new(capacity)),
            DType::Primitive(..) => Self::Int64(Int64Builder::new(capacity)),
            DType::Struct(struct_fields, _) => {
                let mut fields = Vec::with_capacity(struct_fields.nfields());
                for (name, field_dtype) in struct_fields.names().iter().zip(struct_fields.fields())
                {
                    names.push(name.clone());
                    let name = names.len() - 1;
                    fields.push((name, Self::new(&field_dtype, names, capacity)?));
                }
                Self::Struct {
                    names: struct_fields.names().clone(),
                    fields,
                    validity: BitBufferMut::with_capacity(capacity),
                }
            }
            DType::List(element, _) => {
                let mut offsets = BufferMut::with_capacity(capacity + 1);
                offsets.push(0);
                Self::List {
                    offsets,
                    elements: Box::new(Self::new(element, names, capacity)?),
                    validity: BitBufferMut::with_capacity(capacity),
                }
            }
            _ => return None,
        })
    }

    /// Returns `None` if the strings do not fit the offsets of one buffer.
    fn finish(self) -> Option<ArrayRef> {
        Some(match self {
            Self::Utf8(builder) => builder.finish()?,
            Self::Int64(builder) => builder.finish(),
            Self::Struct {
                names,
                fields,
                validity,
            } => {
                let len = validity.len();
                let fields = fields
                    .into_iter()
                    .map(|(_, field)| field.finish())
                    .collect::<Option<Vec<_>>>()?;
                StructArray::try_new(names, fields, len, Validity::from(validity.freeze()))
                    .ok()?
                    .into_array()
            }
            Self::List {
                offsets,
                elements,
                validity,
            } => {
                let elements = elements.finish()?;
                let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable);
                // SAFETY: the offsets start at zero and grow by the number of elements each list
                // appended, so they end at the number of elements, and there is one more offset
                // than there are validity bits.
                unsafe {
                    ListArray::new_unchecked(
                        elements,
                        offsets.into_array(),
                        Validity::from(validity.freeze()),
                    )
                }
                .into_array()
            }
        })
    }
}

impl Append for Builder {
    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Option<()> {
        match self {
            Self::Utf8(builder) => builder.append(fields, value),
            Self::Int64(builder) => builder.append(fields, value),
            Self::Struct {
                fields: builders,
                validity,
                ..
            } => {
                let object = Object::new(value)?;
                for (name, builder) in builders.iter_mut() {
                    // Every field of a value that is not an object is null, as is the struct.
                    let field = match &object {
                        Some(object) => fields.find(object, *name)?,
                        None => None,
                    };
                    match field {
                        Some(field) => builder.append(fields, field)?,
                        None => builder.append_null(),
                    }
                }
                validity.append(object.is_some());
                Some(())
            }
            Self::List {
                offsets,
                elements,
                validity,
            } => {
                let list = Elements::new(value)?;
                let len = list.as_ref().map_or(0, |list| list.len);
                for idx in 0..len {
                    let element = list.as_ref().and_then(|list| list.element(idx))?;
                    elements.append(fields, element)?;
                }
                let end = offsets.last().copied().unwrap_or_default();
                offsets.push(end + u64::try_from(len).ok()?);
                validity.append(list.is_some());
                Some(())
            }
        }
    }

    fn append_null(&mut self) {
        match self {
            Self::Utf8(builder) => builder.append_null(),
            Self::Int64(builder) => builder.append_null(),
            Self::Struct {
                fields, validity, ..
            } => {
                for (_, field) in fields {
                    field.append_null();
                }
                validity.append(false);
            }
            Self::List {
                offsets, validity, ..
            } => {
                let end = offsets.last().copied().unwrap_or_default();
                offsets.push(end);
                validity.append(false);
            }
        }
    }
}

struct Utf8Builder {
    views: BufferMut<BinaryView>,
    bytes: ByteBufferMut,
    validity: BitBufferMut,
}

impl Utf8Builder {
    fn new(capacity: usize) -> Self {
        Self {
            views: BufferMut::with_capacity(capacity),
            bytes: ByteBufferMut::empty(),
            validity: BitBufferMut::with_capacity(capacity),
        }
    }

    #[inline]
    fn push(&mut self, string: Option<&str>) {
        self.validity.append(string.is_some());
        let string = string.unwrap_or_default().as_bytes();
        // The views are built here rather than by a builder, which would check the UTF-8 of every
        // string a second time. The offset of an inlined string is not read, and `finish` checks
        // that the offsets of the others fit.
        let offset = u32::try_from(self.bytes.len()).unwrap_or(u32::MAX);
        if string.len() > BinaryView::MAX_INLINED_SIZE {
            self.bytes.extend_from_slice(string);
        }
        self.views.push(BinaryView::make_view(string, 0, offset));
    }

    /// Returns `None` if the strings do not fit the offsets of one buffer.
    fn finish(self) -> Option<ArrayRef> {
        u32::try_from(self.bytes.len()).ok()?;
        // SAFETY: every view is either inlined or spans a string just copied to the only buffer,
        // and each string was checked to be UTF-8 when it was read.
        let array = unsafe {
            VarBinViewArray::new_unchecked(
                self.views.freeze(),
                Arc::from([self.bytes.freeze()]),
                DType::Utf8(Nullability::Nullable),
                Validity::from(self.validity.freeze()),
            )
        };
        Some(array.into_array())
    }
}

impl Append for Utf8Builder {
    #[inline]
    fn append(&mut self, _fields: &Fields<'_>, value: &[u8]) -> Option<()> {
        let header = *value.first()?;
        let payload = &value[1..];
        let string = match (header & BASIC_TYPE_MASK, header >> 2) {
            (SHORT_STRING, len) => Some(utf8(payload.get(..usize::from(len))?)?),
            (PRIMITIVE, LONG_STRING) => {
                let len = read_uint(payload, 4)?;
                Some(utf8(payload.get(4..4 + len)?)?)
            }
            (OBJECT | ARRAY, _) => None,
            // What `Variant::as_string` returns, which borrows from the variant and not from the
            // value.
            _ => match decode(value)? {
                Variant::String(string) => Some(string),
                Variant::ShortString(string) => Some(string.as_str()),
                _ => None,
            },
        };
        self.push(string);
        Some(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.push(None);
    }
}

struct Int64Builder {
    values: BufferMut<i64>,
    validity: BitBufferMut,
}

impl Int64Builder {
    fn new(capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            validity: BitBufferMut::with_capacity(capacity),
        }
    }

    #[inline]
    fn push(&mut self, integer: Option<i64>) {
        self.values.push(integer.unwrap_or_default());
        self.validity.append(integer.is_some());
    }

    fn finish(self) -> ArrayRef {
        PrimitiveArray::new(self.values.freeze(), Validity::from(self.validity.freeze()))
            .into_array()
    }
}

impl Append for Int64Builder {
    #[inline]
    fn append(&mut self, _fields: &Fields<'_>, value: &[u8]) -> Option<()> {
        let header = *value.first()?;
        let payload = &value[1..];
        let integer = match (header & BASIC_TYPE_MASK, header >> 2) {
            (PRIMITIVE, INT8) => Some(i64::from(i8::from_le_bytes(array(payload)?))),
            (PRIMITIVE, INT16) => Some(i64::from(i16::from_le_bytes(array(payload)?))),
            (PRIMITIVE, INT32) => Some(i64::from(i32::from_le_bytes(array(payload)?))),
            (PRIMITIVE, INT64) => Some(i64::from_le_bytes(array(payload)?)),
            (OBJECT | ARRAY, _) => None,
            _ => decode(value)?.as_int64(),
        };
        self.push(integer);
        Some(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.push(None);
    }
}

fn utf8(bytes: &[u8]) -> Option<&str> {
    std::str::from_utf8(bytes).ok()
}

/// The bytes of row `idx`, borrowed from `array`.
pub(crate) fn bytes_at(array: &VarBinViewArray, idx: usize) -> &[u8] {
    let view = &array.views()[idx];
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &array.buffer(view.buffer_index as usize).as_slice()[view.as_range()]
    }
}

/// How to find the fields of the cached names in the objects of rows that share one metadata.
#[derive(Clone, Copy)]
pub(crate) enum Fields<'a> {
    /// The dictionary id of each name. [`ABSENT`] is the id of a name that the dictionary does
    /// not hold, which no object has.
    Ids(&'a [usize]),
    /// The dictionary holds one of the names more than once, so that the name has no single id.
    /// The fields are found by name.
    Names(Metadata<'a>, &'a [FieldName]),
}

impl Fields<'_> {
    /// The field of `object` with the cached name at index `name`, or `Some(None)` if the object
    /// has no such field.
    #[inline]
    fn find<'v>(&self, object: &Object<'v>, name: usize) -> Option<Option<&'v [u8]>> {
        match self {
            Self::Ids(ids) => object.field_with_id(ids[name]),
            Self::Names(metadata, names) => object.field(metadata, names[name].as_ref()),
        }
    }

    /// The position in `object` of the field with the cached name at index `name`, or
    /// `Some(None)` if the object has no such field.
    #[inline]
    pub(crate) fn position(&self, object: &Object<'_>, name: usize) -> Option<Option<usize>> {
        match self {
            Self::Ids(ids) => object.position_with_id(ids[name]),
            Self::Names(metadata, names) => object.position(metadata, names[name].as_ref()),
        }
    }
}

/// An id that is not in any dictionary: the ids of an object are at most four bytes wide.
const ABSENT: usize = usize::MAX;

/// The [`Fields`] of a list of names, for the metadata of the rows seen most recently.
///
/// Few rows have a metadata of their own: the rows of a table share a handful of shapes, and
/// the metadata child is dictionary encoded. Resolving the names against a metadata once turns
/// the lookup in every object from a binary search over names into a scan for an id.
///
/// The slots are direct mapped on a hash of the metadata's view. Rows with the same metadata
/// under a dictionary have the same view. A miss writes the ids of the slot in place, so that it
/// neither allocates nor moves them.
pub(crate) struct FieldCache<'a> {
    names: &'a [FieldName],
    /// The view of the metadata that each slot holds.
    keys: Vec<Option<u128>>,
    /// The ids of the names for each slot, one run of `names.len()` ids per slot.
    ids: Vec<usize>,
    /// The metadata of each slot whose dictionary holds one of the names more than once.
    duplicates: Vec<Option<Metadata<'a>>>,
}

impl<'a> FieldCache<'a> {
    const SLOTS: usize = 256;

    pub(crate) fn new(names: &'a [FieldName]) -> Self {
        Self {
            names,
            keys: vec![None; Self::SLOTS],
            ids: vec![ABSENT; Self::SLOTS * names.len()],
            duplicates: vec![None; Self::SLOTS],
        }
    }

    /// The fields of the names for row `idx` of `metadata`.
    #[inline]
    pub(crate) fn resolve(
        &mut self,
        metadata: &'a VarBinViewArray,
        idx: usize,
    ) -> Option<Fields<'_>> {
        let key = metadata.views()[idx].as_u128();
        // The top byte of the low half of a multiplicative hash, as the table has 256 slots.
        let mixed = (key ^ (key >> 64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let slot = usize::from(mixed.to_le_bytes()[7]);
        let len = self.names.len();
        let ids = &mut self.ids[slot * len..(slot + 1) * len];

        if self.keys[slot] != Some(key) {
            // The slot is empty until its ids are complete, as a metadata that does not decode
            // stops the fill part of the way through.
            self.keys[slot] = None;
            self.duplicates[slot] = Self::fill(self.names, ids, bytes_at(metadata, idx))?;
            self.keys[slot] = Some(key);
        }

        Some(match self.duplicates[slot] {
            Some(metadata) => Fields::Names(metadata, self.names),
            None => Fields::Ids(ids),
        })
    }

    /// Writes the dictionary id of each name in `names` to `ids`.
    ///
    /// Returns the metadata if its dictionary holds one of the names more than once.
    fn fill(
        names: &[FieldName],
        ids: &mut [usize],
        metadata: &'a [u8],
    ) -> Option<Option<Metadata<'a>>> {
        let metadata = Metadata::new(metadata)?;

        // A path or a small struct has a number of names known when the scan compiles, so that
        // the loop over the names unrolls.
        if let Ok(ids) = <&mut [usize; 1]>::try_from(&mut *ids) {
            return Self::scan(names, ids, metadata);
        }
        if let Ok(ids) = <&mut [usize; 2]>::try_from(&mut *ids) {
            return Self::scan(names, ids, metadata);
        }
        if let Ok(ids) = <&mut [usize; 3]>::try_from(&mut *ids) {
            return Self::scan(names, ids, metadata);
        }
        Self::scan(names, ids, metadata)
    }

    /// Generic over the ids, so that each array length of [`Self::fill`] compiles a scan of
    /// its own.
    fn scan(
        names: &[FieldName],
        mut ids: impl AsMut<[usize]>,
        metadata: Metadata<'a>,
    ) -> Option<Option<Metadata<'a>>> {
        let ids = ids.as_mut();
        ids.fill(ABSENT);

        // Every row of a table without a dictionary over its metadata misses, so this loop is
        // hot. Each offset is read once, and the bytes of a name are compared only with names of
        // the same length.
        let mut offsets = metadata.offsets.chunks_exact(metadata.offset_size);
        let mut start = read_uint(offsets.next()?, metadata.offset_size)?;
        for (id, end) in offsets.take(metadata.len).enumerate() {
            let end = read_uint(end, metadata.offset_size)?;
            let len = end.checked_sub(start)?;
            for (field, found) in names.iter().zip(ids.iter_mut()) {
                let field = field.as_ref().as_bytes();
                if field.len() == len && metadata.names.get(start..end)? == field {
                    if *found != ABSENT {
                        return Some(Some(metadata));
                    }
                    *found = id;
                }
            }
            start = end;
        }
        Some(None)
    }
}

/// The first `N` bytes of `bytes`.
#[inline]
pub(crate) fn array<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    bytes.get(..N)?.try_into().ok()
}

/// Reads a little-endian unsigned integer of `width` bytes, at most four.
#[inline]
pub(crate) fn read_uint(bytes: &[u8], width: usize) -> Option<usize> {
    // The width is one byte far more often than not, and a fixed-width read is a single load
    // where copying a variable number of bytes is a call.
    Some(match width {
        1 => usize::from(*bytes.first()?),
        2 => usize::from(u16::from_le_bytes(array(bytes)?)),
        3 => {
            let [low, mid, high] = array(bytes)?;
            usize::from(low) | usize::from(mid) << 8 | usize::from(high) << 16
        }
        4 => usize::try_from(u32::from_le_bytes(array(bytes)?)).ok()?,
        _ => return None,
    })
}

/// The two low bits of a value's first byte, its basic type.
pub(crate) const BASIC_TYPE_MASK: u8 = 0b11;
pub(crate) const PRIMITIVE: u8 = 0;
pub(crate) const SHORT_STRING: u8 = 1;
pub(crate) const OBJECT: u8 = 2;
pub(crate) const ARRAY: u8 = 3;

/// Primitive type ids, the six high bits of a primitive value's first byte.
pub(crate) const INT8: u8 = 3;
pub(crate) const INT16: u8 = 4;
pub(crate) const INT32: u8 = 5;
pub(crate) const INT64: u8 = 6;
pub(crate) const LONG_STRING: u8 = 16;

/// The dictionary of field names of a Variant value.
#[derive(Clone, Copy)]
pub(crate) struct Metadata<'a> {
    /// The number of names.
    len: usize,
    /// The width in bytes of an offset.
    offset_size: usize,
    /// The offsets of the names, one more than there are names.
    offsets: &'a [u8],
    /// The names, back to back.
    names: &'a [u8],
}

impl<'a> Metadata<'a> {
    fn new(bytes: &'a [u8]) -> Option<Self> {
        let header = *bytes.first()?;
        let offset_size = usize::from(header >> 6) + 1;
        let len = read_uint(bytes.get(1..)?, offset_size)?;
        let offsets_start = 1 + offset_size;
        let names_start = offsets_start + (len + 1) * offset_size;
        Some(Self {
            len,
            offset_size,
            offsets: bytes.get(offsets_start..names_start)?,
            names: bytes.get(names_start..)?,
        })
    }

    /// The name with dictionary id `id`.
    fn name(&self, id: usize) -> Option<&'a [u8]> {
        let start = read_uint(self.offsets.get(id * self.offset_size..)?, self.offset_size)?;
        let end = read_uint(
            self.offsets.get((id + 1) * self.offset_size..)?,
            self.offset_size,
        )?;
        self.names.get(start..end)
    }
}

/// An object value: its field ids, sorted by field name, and the offsets of the field values.
pub(crate) struct Object<'a> {
    len: usize,
    id_size: usize,
    offset_size: usize,
    ids: &'a [u8],
    offsets: &'a [u8],
    values: &'a [u8],
}

impl<'a> Object<'a> {
    /// Reads the object header of `value`, or returns `Some(None)` if it is not an object.
    pub(crate) fn new(value: &'a [u8]) -> Option<Option<Self>> {
        let header = *value.first()?;
        if header & BASIC_TYPE_MASK != OBJECT {
            return Some(None);
        }
        let header = header >> 2;
        let offset_size = usize::from(header & 0b11) + 1;
        let id_size = usize::from((header >> 2) & 0b11) + 1;
        let len_size = if header & 0b1_0000 != 0 { 4 } else { 1 };

        let len = read_uint(value.get(1..)?, len_size)?;
        let ids_start = 1 + len_size;
        let offsets_start = ids_start + len * id_size;
        let values_start = offsets_start + (len + 1) * offset_size;
        Some(Some(Self {
            len,
            id_size,
            offset_size,
            ids: value.get(ids_start..offsets_start)?,
            offsets: value.get(offsets_start..values_start)?,
            values: value.get(values_start..)?,
        }))
    }

    /// The number of fields.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The width in bytes of a field id.
    pub(crate) fn id_size(&self) -> usize {
        self.id_size
    }

    /// The id bytes of the field at `position`, as the object stores them.
    pub(crate) fn id_bytes(&self, position: usize) -> Option<&'a [u8]> {
        self.ids
            .get(position * self.id_size..(position + 1) * self.id_size)
    }

    /// The value of the field at `position`, followed by whatever the object stores after it.
    #[inline]
    pub(crate) fn value(&self, position: usize) -> Option<&'a [u8]> {
        let offset = read_uint(
            self.offsets.get(position * self.offset_size..)?,
            self.offset_size,
        )?;
        self.values.get(offset..)
    }

    /// The position of the field whose name has dictionary id `id`, or `Some(None)` if the
    /// object has no such field.
    #[inline]
    fn position_with_id(&self, id: usize) -> Option<Option<usize>> {
        if self.id_size == 1 {
            return Some(self.ids.iter().position(|&field| usize::from(field) == id));
        }
        for (idx, field) in self.ids.chunks_exact(self.id_size).enumerate() {
            if read_uint(field, self.id_size)? == id {
                return Some(Some(idx));
            }
        }
        Some(None)
    }

    /// The value of the field whose name has dictionary id `id`, or `Some(None)` if the object
    /// has no such field.
    #[inline]
    fn field_with_id(&self, id: usize) -> Option<Option<&'a [u8]>> {
        match self.position_with_id(id)? {
            Some(position) => self.value(position).map(Some),
            None => Some(None),
        }
    }

    /// The position of the field `name`, or `Some(None)` if the object has no such field.
    fn position(&self, metadata: &Metadata<'_>, name: &str) -> Option<Option<usize>> {
        let (mut low, mut high) = (0, self.len);
        while low < high {
            let mid = low + (high - low) / 2;
            let id = read_uint(self.ids.get(mid * self.id_size..)?, self.id_size)?;
            match metadata.name(id)?.cmp(name.as_bytes()) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(Some(mid)),
            }
        }
        Some(None)
    }

    /// The value of the field `name`, or `Some(None)` if the object has no such field.
    fn field(&self, metadata: &Metadata<'_>, name: &str) -> Option<Option<&'a [u8]>> {
        match self.position(metadata, name)? {
            Some(position) => self.value(position).map(Some),
            None => Some(None),
        }
    }
}

/// An array value: the offsets of its elements, which follow one another.
pub(crate) struct Elements<'a> {
    len: usize,
    offset_size: usize,
    offsets: &'a [u8],
    values: &'a [u8],
}

impl<'a> Elements<'a> {
    /// Reads the array header of `value`, or returns `Some(None)` if it is not an array.
    pub(crate) fn new(value: &'a [u8]) -> Option<Option<Self>> {
        let header = *value.first()?;
        if header & BASIC_TYPE_MASK != ARRAY {
            return Some(None);
        }
        let header = header >> 2;
        let offset_size = usize::from(header & 0b11) + 1;
        let len_size = if header & 0b100 != 0 { 4 } else { 1 };

        let len = read_uint(value.get(1..)?, len_size)?;
        let offsets_start = 1 + len_size;
        let values_start = offsets_start + (len + 1) * offset_size;
        Some(Some(Self {
            len,
            offset_size,
            offsets: value.get(offsets_start..values_start)?,
            values: value.get(values_start..)?,
        }))
    }

    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The value of element `idx`, followed by the elements after it.
    pub(crate) fn element(&self, idx: usize) -> Option<&'a [u8]> {
        let offset = read_uint(
            self.offsets.get(idx * self.offset_size..)?,
            self.offset_size,
        )?;
        self.values.get(offset..)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StringArray;
    use parquet_variant::VariantPath as PqVariantPath;
    use parquet_variant_compute::GetOptions;
    use parquet_variant_compute::json_to_variant;
    use parquet_variant_compute::variant_get;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::StructFields;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ParquetVariant;
    use crate::ParquetVariantArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// A string too long for the short string encoding, which holds at most 63 bytes.
    fn long_string() -> String {
        "x".repeat(100)
    }

    /// An object with more fields than the one-byte count of a small object holds.
    fn large_object() -> String {
        let fields = (0..300)
            .map(|idx| format!(r#""f{idx:03}":{idx}"#))
            .collect::<Vec<_>>();
        format!("{{{}}}", fields.join(","))
    }

    /// Extracts `path` from `rows` as `dtype` in place, and with the general kernel.
    ///
    /// Returns the in-place result, if the extraction is covered, and the kernel's.
    fn extract_both(
        rows: &[String],
        path: &str,
        dtype: &DType,
    ) -> VortexResult<(Option<ArrayRef>, ArrayRef)> {
        let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values(rows));
        let arrow_variant = json_to_variant(&json)?;
        let arrow_storage: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
        let field = Arc::new(SESSION.arrow().to_arrow_field("", dtype)?);
        let pq_path = if path.is_empty() {
            PqVariantPath::default()
        } else {
            PqVariantPath::try_from(path)?
        };
        let expected = variant_get(
            &arrow_storage,
            GetOptions::new_with_path(pq_path).with_as_type(Some(Arc::clone(&field))),
        )?;
        let expected = SESSION.arrow().from_arrow_array(expected, &field)?;

        let storage = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        let storage = storage.as_::<ParquetVariant>();
        let mut ctx = SESSION.create_execution_ctx();
        let metadata = storage
            .metadata()
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?;
        let value = storage
            .value()
            .ok_or_else(|| vortex_err!("unshredded storage has a value"))?
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?;
        let path = if path.is_empty() {
            vec![]
        } else {
            path.split('.').collect::<Vec<_>>()
        };
        let actual = extract(&metadata, &value, &Mask::new_true(rows.len()), &path, dtype);
        Ok((actual, expected))
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::Nullable)
    }

    fn int64() -> DType {
        DType::Primitive(PType::I64, Nullability::Nullable)
    }

    fn struct_of<const N: usize>(fields: [(&str, DType); N]) -> DType {
        DType::Struct(StructFields::from_iter(fields), Nullability::Nullable)
    }

    fn list_of(element: DType) -> DType {
        DType::List(Arc::new(element), Nullability::Nullable)
    }

    /// A struct or a list is extracted as the general kernel extracts it, from rows whose values
    /// at the path are objects, arrays, and neither, with fields and elements of every type.
    #[rstest]
    #[case::struct_at_the_root(
        "",
        struct_of([
            ("kind", utf8()),
            ("time_us", int64()),
            ("commit", struct_of([("collection", utf8()), ("rev", int64())])),
        ])
    )]
    #[case::struct_at_a_path("commit", struct_of([("collection", utf8()), ("rev", int64()), ("absent", utf8())]))]
    #[case::list_of_strings("tags", list_of(utf8()))]
    #[case::list_of_integers("nums", list_of(int64()))]
    #[case::list_of_structs("items", list_of(struct_of([("n", int64())])))]
    #[case::list_in_a_struct("", struct_of([("tags", list_of(utf8())), ("kind", utf8())]))]
    fn extracts_structs_and_lists_as_the_kernel_does(
        #[case] path: &str,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let rows = vec![
            r#"{"kind":"commit","time_us":5,"commit":{"collection":"post","rev":1},"tags":["a","b"],"nums":[1,2.5,"x",null],"items":[{"n":1},{"n":"two"},3]}"#.to_string(),
            r#"{"kind":"identity","commit":"not an object","tags":"not a list","nums":[],"items":[]}"#.to_string(),
            r#"{"kind":null,"commit":{"rev":2.0},"tags":[null,"c",{"x":1}],"nums":[300000000000]}"#.to_string(),
            r#"{"other":true}"#.to_string(),
            r#""a string""#.to_string(),
            "[1,2,3]".to_string(),
            large_object(),
        ];

        let (actual, expected) = extract_both(&rows, path, &dtype)?;

        let actual = actual.ok_or_else(|| vortex_err!("{path} as {dtype} must be covered"))?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A value that does not convert is a null, so non-nullable fields and elements are not
    /// covered, and neither are the dtypes the general kernel covers on its own.
    #[rstest]
    #[case::non_nullable_field(DType::Struct(
        StructFields::from_iter([("a", DType::Utf8(Nullability::NonNullable))]),
        Nullability::Nullable,
    ))]
    #[case::non_nullable_element(DType::List(
        Arc::new(DType::Utf8(Nullability::NonNullable)),
        Nullability::Nullable,
    ))]
    #[case::uncovered_field(struct_of([("a", DType::Bool(Nullability::Nullable))]))]
    #[case::struct_without_fields(struct_of([]))]
    fn does_not_cover_non_nullable_or_uncovered_children(#[case] dtype: DType) {
        assert!(!covers(&dtype));
    }

    /// Whatever is extracted in place is what the general kernel extracts.
    #[rstest]
    #[case::field("kind", utf8())]
    #[case::nested("commit.collection", utf8())]
    #[case::deeply_nested("commit.record.text", utf8())]
    #[case::integer("time_us", int64())]
    #[case::nested_integer("commit.rev", int64())]
    #[case::missing("absent", utf8())]
    #[case::missing_nested("commit.absent", int64())]
    #[case::field_of_a_large_object("f123", int64())]
    fn extracts_what_the_kernel_extracts(
        #[case] path: &str,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let rows = vec![
            // Short and long strings, and integers of every width.
            format!(
                r#"{{"kind":"commit","time_us":1732206349000167,"commit":{{"collection":"post","rev":1,"record":{{"text":"{}"}}}}}}"#,
                long_string()
            ),
            r#"{"kind":"identity","time_us":-70000,"commit":{"collection":"","rev":-300}}"#
                .to_string(),
            r#"{"kind":"account","time_us":7,"commit":{"rev":3000000000}}"#.to_string(),
            // An object without the field, a field that is not an object, and values that
            // are not objects.
            r#"{"other":true}"#.to_string(),
            r#"{"kind":"commit","commit":"not an object"}"#.to_string(),
            r#""a string""#.to_string(),
            "[1,2,3]".to_string(),
            large_object(),
        ];

        let (actual, expected) = extract_both(&rows, path, &dtype)?;

        let actual = actual.ok_or_else(|| vortex_err!("{path} as {dtype} must be covered"))?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A value of another type than the one asked for converts as the general kernel converts
    /// it, so a row of another type does not turn the extraction down.
    #[rstest]
    #[case::integer_as_string(r#"{"a":1}"#, utf8())]
    #[case::double_as_string(r#"{"a":1.5}"#, utf8())]
    #[case::boolean_as_string(r#"{"a":true}"#, utf8())]
    #[case::null_as_string(r#"{"a":null}"#, utf8())]
    #[case::object_as_string(r#"{"a":{"b":"x"}}"#, utf8())]
    #[case::array_as_string(r#"{"a":["x"]}"#, utf8())]
    #[case::string_as_integer(r#"{"a":"1"}"#, int64())]
    #[case::null_as_integer(r#"{"a":null}"#, int64())]
    #[case::boolean_as_integer(r#"{"a":true}"#, int64())]
    #[case::fraction_as_integer(r#"{"a":1.5}"#, int64())]
    #[case::whole_fraction_as_integer(r#"{"a":12.00}"#, int64())]
    #[case::negative_fraction_as_integer(r#"{"a":-7.25}"#, int64())]
    #[case::out_of_range_as_integer(r#"{"a":1e300}"#, int64())]
    #[case::object_as_integer(r#"{"a":{"b":1}}"#, int64())]
    #[case::array_as_integer(r#"{"a":[1]}"#, int64())]
    fn converts_as_the_kernel_does(#[case] row: &str, #[case] dtype: DType) -> VortexResult<()> {
        // Rows that are read without a conversion around the one that needs it.
        let plain = if matches!(dtype, DType::Utf8(_)) {
            r#"{"a":"x"}"#
        } else {
            r#"{"a":5}"#
        };
        let rows = vec![plain.to_string(), row.to_string(), plain.to_string()];

        let (actual, expected) = extract_both(&rows, "a", &dtype)?;

        let actual = actual.ok_or_else(|| vortex_err!("{row} as {dtype} must be covered"))?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A dictionary may hold a name twice unless it is sorted, and an object may then use
    /// either id.
    #[rstest]
    #[case::first_id(0)]
    #[case::second_id(1)]
    fn finds_a_field_whose_name_is_in_the_dictionary_twice(#[case] id: u8) -> VortexResult<()> {
        // Version 1 with one-byte offsets, then two names, both "a".
        let metadata = [0x01, 2, 0, 1, 2, b'a', b'a'];
        // An object of one field, whose value is the short string "x".
        let value = [0x02, 1, id, 0, 2, 0x05, b'x'];
        let extracted = extract(
            &VarBinViewArray::from_iter_bin([metadata]),
            &VarBinViewArray::from_iter_bin([value]),
            &Mask::new_true(1),
            &["a"],
            &utf8(),
        )
        .ok_or_else(|| vortex_err!("not extracted in place"))?;
        assert_arrays_eq!(
            extracted,
            VarBinViewArray::from_iter_nullable_str([Some("x")]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// Rows that hold no value are null, whatever their bytes are.
    #[test]
    fn rows_without_a_value_are_null() -> VortexResult<()> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(vec![r#"{"a":"x"}"#, r#"{"a":"y"}"#]));
        let storage =
            ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())?;
        let storage = storage.as_::<ParquetVariant>();
        let mut ctx = SESSION.create_execution_ctx();
        let metadata = storage
            .metadata()
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?;
        let value = storage
            .value()
            .ok_or_else(|| vortex_err!("unshredded storage has a value"))?
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?;

        let actual = extract(
            &metadata,
            &value,
            &Mask::from_iter([true, false]),
            &["a"],
            &utf8(),
        )
        .ok_or_else(|| vortex_err!("a string field must be covered"))?;

        assert_arrays_eq!(
            actual,
            VarBinViewArray::from_iter_nullable_str([Some("x"), None]),
            &mut ctx
        );
        Ok(())
    }
}
