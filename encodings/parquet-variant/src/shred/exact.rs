// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shreds unshredded storage by reading the [Variant encoding] in place.
//!
//! A value moves into a typed column only if its type is the type of the column, as the
//! [Variant shredding] specification requires. Every other value stays in the residual binary
//! value, byte for byte. The only conversion is between the integer widths, which are one
//! equivalence class: an integer moves into an integer column that it fits.
//!
//! An object moves into a shredded struct, and the residual value of the row keeps the fields
//! that the struct does not shred. An array moves into a shredded list, and each element is
//! shredded by the element of the schema.
//!
//! [Variant encoding]: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
//! [Variant shredding]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::extract::ARRAY;
use crate::extract::BASIC_TYPE_MASK;
use crate::extract::Elements;
use crate::extract::FieldCache;
use crate::extract::Fields;
use crate::extract::INT8;
use crate::extract::INT16;
use crate::extract::INT32;
use crate::extract::INT64;
use crate::extract::LONG_STRING;
use crate::extract::OBJECT;
use crate::extract::Object;
use crate::extract::PRIMITIVE;
use crate::extract::SHORT_STRING;
use crate::extract::array;
use crate::extract::bytes_at;
use crate::extract::read_uint;

/// Primitive type ids that [`crate::extract`] does not read.
const TRUE: u8 = 1;
const FALSE: u8 = 2;
const DOUBLE: u8 = 7;
const DECIMAL4: u8 = 8;
const DECIMAL8: u8 = 9;
const DECIMAL16: u8 = 10;
const DATE: u8 = 11;
const TIMESTAMP_MICROS: u8 = 12;
const TIMESTAMP_NTZ_MICROS: u8 = 13;
const FLOAT: u8 = 14;
const BINARY: u8 = 15;
const TIME_NTZ: u8 = 17;
const TIMESTAMP_NANOS: u8 = 18;
const TIMESTAMP_NTZ_NANOS: u8 = 19;
const UUID: u8 = 20;

/// Shreds the values of unshredded storage by `schema`.
///
/// `metadata` and `value` are the storage children, and `valid` the rows that hold a value.
/// Returns the residual `value` child and the `typed_value` child of the shredded storage.
pub(super) fn shred(
    metadata: &VarBinViewArray,
    value: &VarBinViewArray,
    valid: &Mask,
    schema: &DType,
    allocator: &BufferAllocatorRef,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let mut names = Vec::new();
    let mut shredder = Shredder::new(schema, &mut names, value.len(), allocator)?;
    let mut cache = FieldCache::new(&names);

    for idx in 0..value.len() {
        if !valid.value(idx) {
            shredder.append_missing();
            continue;
        }

        let malformed = || vortex_err!("Variant value of row {idx} is malformed");
        let fields = cache.resolve(metadata, idx).ok_or_else(malformed)?;
        shredder
            .append(&fields, bytes_at(value, idx))
            .ok_or_else(malformed)?;
    }

    shredder.finish()
}

/// The `value` and `typed_value` columns of one node of the schema.
struct Shredder {
    /// The binary values that are not in `typed`.
    value: VarBinViewBuilder,
    typed: Typed,
}

/// The builder of a `typed_value` column.
enum Typed {
    Utf8(VarBinViewBuilder),
    Binary(VarBinViewBuilder),
    Bool(Leaf<bool>),
    I8(Leaf<i8>),
    I16(Leaf<i16>),
    I32(Leaf<i32>),
    I64(Leaf<i64>),
    F32(Leaf<f32>),
    F64(Leaf<f64>),
    Object(Box<ObjectShredder>),
    List(Box<ListShredder>),
}

/// A shredded struct. Each field shreds the field of the object of the same name.
struct ObjectShredder {
    names: FieldNames,
    /// The shredder of each field, with the index of its name in the [`FieldCache`].
    fields: Vec<(usize, Shredder)>,
    validity: BitBufferMut,
    /// For each field of the current object, whether a shredder took it.
    taken: Vec<bool>,
    /// The residual object of the current row.
    residual: Vec<u8>,
}

/// A shredded list. Each element is shredded by the same shredder.
struct ListShredder {
    offsets: BufferMut<u64>,
    elements: Shredder,
    validity: BitBufferMut,
}

/// The values and validity of a typed leaf.
struct Leaf<T> {
    values: BufferMut<T>,
    validity: BitBufferMut,
}

impl<T: Copy + Default> Leaf<T> {
    fn new(capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            validity: BitBufferMut::with_capacity(capacity),
        }
    }

    #[inline]
    fn push(&mut self, value: Option<T>) {
        self.values.push(value.unwrap_or_default());
        self.validity.append(value.is_some());
    }
}

impl<T: NativePType> Leaf<T> {
    fn finish(self) -> ArrayRef {
        PrimitiveArray::new(self.values.freeze(), Validity::from(self.validity.freeze()))
            .into_array()
    }
}

impl Leaf<bool> {
    fn finish_bool(self) -> ArrayRef {
        let values = self.values.iter().copied().collect::<BitBufferMut>();
        BoolArray::new(values.freeze(), Validity::from(self.validity.freeze())).into_array()
    }
}

impl Shredder {
    /// A shredder of `schema`, which adds the field names of the structs in `schema` to
    /// `names`. The strings and binaries are allocated by `allocator`.
    fn new(
        schema: &DType,
        names: &mut Vec<FieldName>,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> VortexResult<Self> {
        let typed = match schema {
            DType::Utf8(_) => Typed::Utf8(VarBinViewBuilder::with_capacity_in(
                DType::Utf8(Nullability::Nullable),
                capacity,
                allocator.clone(),
            )),
            DType::Binary(_) => Typed::Binary(binary_builder(capacity, allocator)),
            DType::Bool(_) => Typed::Bool(Leaf::new(capacity)),
            DType::Primitive(PType::I8, _) => Typed::I8(Leaf::new(capacity)),
            DType::Primitive(PType::I16, _) => Typed::I16(Leaf::new(capacity)),
            DType::Primitive(PType::I32, _) => Typed::I32(Leaf::new(capacity)),
            DType::Primitive(PType::I64, _) => Typed::I64(Leaf::new(capacity)),
            DType::Primitive(PType::F32, _) => Typed::F32(Leaf::new(capacity)),
            DType::Primitive(PType::F64, _) => Typed::F64(Leaf::new(capacity)),
            DType::Struct(struct_fields, _) => {
                let mut fields = Vec::with_capacity(struct_fields.nfields());
                for (name, field) in struct_fields.names().iter().zip(struct_fields.fields()) {
                    names.push(name.clone());
                    let name = names.len() - 1;
                    fields.push((name, Self::new(&field, names, capacity, allocator)?));
                }
                Typed::Object(Box::new(ObjectShredder {
                    names: struct_fields.names().clone(),
                    fields,
                    validity: BitBufferMut::with_capacity(capacity),
                    taken: Vec::new(),
                    residual: Vec::new(),
                }))
            }
            DType::List(element, _) => {
                let mut offsets = BufferMut::with_capacity(capacity + 1);
                offsets.push(0);
                Typed::List(Box::new(ListShredder {
                    offsets,
                    elements: Self::new(element, names, capacity, allocator)?,
                    validity: BitBufferMut::with_capacity(capacity),
                }))
            }
            _ => vortex_bail!(
                "a Parquet Variant can only be shredded into strings, binaries, booleans, signed \
                 integers, floats, structs and lists, not {schema}"
            ),
        };

        Ok(Self {
            value: binary_builder(capacity, allocator),
            typed,
        })
    }

    /// Appends a row without a value: an object field that the object does not have, or a
    /// row that is null.
    fn append_missing(&mut self) {
        self.value.append_null();
        self.typed.append_null();
    }

    /// Appends `value`, the bytes of a value followed by any other bytes. Returns `None` for
    /// bytes that are cut short or do not decode.
    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Option<()> {
        let value = value.get(..value_len(value)?)?;

        let taken = match &mut self.typed {
            Typed::Utf8(builder) => {
                let string = string(value);
                if let Some(string) = string {
                    builder.append_value(string);
                }
                string.is_some()
            }
            Typed::Binary(builder) => {
                let binary = binary(value);
                if let Some(binary) = binary {
                    builder.append_value(binary);
                }
                binary.is_some()
            }
            Typed::Bool(leaf) => push(leaf, boolean(value)),
            Typed::I8(leaf) => push(leaf, integer(value).and_then(|v| i8::try_from(v).ok())),
            Typed::I16(leaf) => push(leaf, integer(value).and_then(|v| i16::try_from(v).ok())),
            Typed::I32(leaf) => push(leaf, integer(value).and_then(|v| i32::try_from(v).ok())),
            Typed::I64(leaf) => push(leaf, integer(value)),
            Typed::F32(leaf) => push(leaf, float(value)),
            Typed::F64(leaf) => push(leaf, double(value)),
            Typed::Object(object) => return object.append(fields, value, &mut self.value),
            Typed::List(list) => return list.append(fields, value, &mut self.value),
        };

        if taken {
            self.value.append_null();
        } else {
            self.typed.append_null();
            self.value.append_value(value);
        }
        Some(())
    }

    /// The `value` and `typed_value` columns.
    fn finish(mut self) -> VortexResult<(ArrayRef, ArrayRef)> {
        let value = self.value.finish_into_varbinview().into_array();
        let typed = match self.typed {
            Typed::Utf8(mut builder) | Typed::Binary(mut builder) => {
                builder.finish_into_varbinview().into_array()
            }
            Typed::Bool(leaf) => leaf.finish_bool(),
            Typed::I8(leaf) => leaf.finish(),
            Typed::I16(leaf) => leaf.finish(),
            Typed::I32(leaf) => leaf.finish(),
            Typed::I64(leaf) => leaf.finish(),
            Typed::F32(leaf) => leaf.finish(),
            Typed::F64(leaf) => leaf.finish(),
            Typed::Object(object) => object.finish()?,
            Typed::List(list) => list.finish()?,
        };
        Ok((value, typed))
    }

    /// The struct of the `value` and `typed_value` columns, as a shredded object field or list
    /// element stores them.
    fn finish_field(self, validity: Validity) -> VortexResult<ArrayRef> {
        let len = self.value.len();
        let (value, typed) = self.finish()?;
        Ok(StructArray::try_new(
            FieldNames::from(["value", "typed_value"]),
            vec![value, typed],
            len,
            validity,
        )?
        .into_array())
    }
}

impl Typed {
    fn append_null(&mut self) {
        match self {
            Self::Utf8(builder) | Self::Binary(builder) => builder.append_null(),
            Self::Bool(leaf) => leaf.push(None),
            Self::I8(leaf) => leaf.push(None),
            Self::I16(leaf) => leaf.push(None),
            Self::I32(leaf) => leaf.push(None),
            Self::I64(leaf) => leaf.push(None),
            Self::F32(leaf) => leaf.push(None),
            Self::F64(leaf) => leaf.push(None),
            Self::Object(object) => object.append_null(),
            Self::List(list) => list.append_null(),
        }
    }
}

impl ObjectShredder {
    fn append_null(&mut self) {
        for (_, field) in &mut self.fields {
            field.append_missing();
        }
        self.validity.append(false);
    }

    /// Appends `value`, the bytes of exactly one value, and its residual to `residual`.
    fn append(
        &mut self,
        fields: &Fields<'_>,
        value: &[u8],
        residual: &mut VarBinViewBuilder,
    ) -> Option<()> {
        let Some(object) = Object::new(value)? else {
            self.append_null();
            residual.append_value(value);
            return Some(());
        };

        self.taken.clear();
        self.taken.resize(object.len(), false);
        let mut shredded = 0;
        for (name, field) in &mut self.fields {
            match fields.position(&object, *name)? {
                Some(position) => {
                    *self.taken.get_mut(position)? = true;
                    shredded += 1;
                    field.append(fields, object.value(position)?)?;
                }
                None => field.append_missing(),
            }
        }
        self.validity.append(true);

        if shredded == object.len() {
            residual.append_null();
        } else if shredded == 0 {
            residual.append_value(value);
        } else {
            write_object(&object, &self.taken, &mut self.residual)?;
            residual.append_value(&self.residual);
        }
        Some(())
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        let len = self.validity.len();
        let fields = self
            .fields
            .into_iter()
            .map(|(_, field)| field.finish_field(Validity::NonNullable))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new(
            self.names,
            fields,
            len,
            Validity::from(self.validity.freeze()),
        )?
        .into_array())
    }
}

impl ListShredder {
    fn append_null(&mut self) {
        let end = self.offsets.last().copied().unwrap_or_default();
        self.offsets.push(end);
        self.validity.append(false);
    }

    /// Appends `value`, the bytes of exactly one value, and its residual to `residual`.
    fn append(
        &mut self,
        fields: &Fields<'_>,
        value: &[u8],
        residual: &mut VarBinViewBuilder,
    ) -> Option<()> {
        let Some(list) = Elements::new(value)? else {
            self.append_null();
            residual.append_value(value);
            return Some(());
        };

        for idx in 0..list.len() {
            self.elements.append(fields, list.element(idx)?)?;
        }
        let end = self.offsets.last().copied().unwrap_or_default();
        self.offsets.push(end + u64::try_from(list.len()).ok()?);
        self.validity.append(true);
        residual.append_null();
        Some(())
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        let elements = self.elements.finish_field(Validity::AllValid)?;
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        // SAFETY: the offsets start at zero and grow by the number of elements each list
        // appended, so they end at the number of elements, and there is one more offset than
        // there are validity bits.
        Ok(unsafe {
            ListArray::new_unchecked(
                elements,
                offsets.into_array(),
                Validity::from(self.validity.freeze()),
            )
        }
        .into_array())
    }
}

fn binary_builder(capacity: usize, allocator: &BufferAllocatorRef) -> VarBinViewBuilder {
    VarBinViewBuilder::with_capacity_in(
        DType::Binary(Nullability::Nullable),
        capacity,
        allocator.clone(),
    )
}

/// Pushes `value` to `leaf`, and returns whether there was one.
#[inline]
fn push<T: Copy + Default>(leaf: &mut Leaf<T>, value: Option<T>) -> bool {
    if value.is_some() {
        leaf.push(value);
    }
    value.is_some()
}

/// The basic type of `value`, and the six bits above it.
#[inline]
fn header(value: &[u8]) -> Option<(u8, u8)> {
    let header = *value.first()?;
    Some((header & BASIC_TYPE_MASK, header >> 2))
}

/// The string that `value` holds, if it is a string.
fn string(value: &[u8]) -> Option<&str> {
    let bytes = match header(value)? {
        (SHORT_STRING, len) => value.get(1..1 + usize::from(len))?,
        (PRIMITIVE, LONG_STRING) => sized(value)?,
        _ => return None,
    };
    std::str::from_utf8(bytes).ok()
}

/// The bytes that `value` holds, if it is a binary.
fn binary(value: &[u8]) -> Option<&[u8]> {
    match header(value)? {
        (PRIMITIVE, BINARY) => sized(value),
        _ => None,
    }
}

/// The payload of a primitive with a four-byte size.
fn sized(value: &[u8]) -> Option<&[u8]> {
    let len = read_uint(value.get(1..)?, 4)?;
    value.get(5..5 + len)
}

fn boolean(value: &[u8]) -> Option<bool> {
    match header(value)? {
        (PRIMITIVE, TRUE) => Some(true),
        (PRIMITIVE, FALSE) => Some(false),
        _ => None,
    }
}

/// The integer that `value` holds, if it is an integer of any width.
fn integer(value: &[u8]) -> Option<i64> {
    let payload = value.get(1..)?;
    match header(value)? {
        (PRIMITIVE, INT8) => Some(i64::from(i8::from_le_bytes(array(payload)?))),
        (PRIMITIVE, INT16) => Some(i64::from(i16::from_le_bytes(array(payload)?))),
        (PRIMITIVE, INT32) => Some(i64::from(i32::from_le_bytes(array(payload)?))),
        (PRIMITIVE, INT64) => Some(i64::from_le_bytes(array(payload)?)),
        _ => None,
    }
}

fn float(value: &[u8]) -> Option<f32> {
    match header(value)? {
        (PRIMITIVE, FLOAT) => Some(f32::from_le_bytes(array(value.get(1..)?)?)),
        _ => None,
    }
}

fn double(value: &[u8]) -> Option<f64> {
    match header(value)? {
        (PRIMITIVE, DOUBLE) => Some(f64::from_le_bytes(array(value.get(1..)?)?)),
        _ => None,
    }
}

/// The length in bytes of the value at the start of `value`.
fn value_len(value: &[u8]) -> Option<usize> {
    let (basic_type, bits) = header(value)?;
    let bits = usize::from(bits);
    match basic_type {
        SHORT_STRING => Some(1 + bits),
        PRIMITIVE => Some(
            1 + match u8::try_from(bits).ok()? {
                0 | TRUE | FALSE => 0,
                INT8 => 1,
                INT16 => 2,
                INT32 | DATE | FLOAT => 4,
                INT64 | DOUBLE | TIMESTAMP_MICROS | TIMESTAMP_NTZ_MICROS | TIME_NTZ
                | TIMESTAMP_NANOS | TIMESTAMP_NTZ_NANOS => 8,
                DECIMAL4 => 5,
                DECIMAL8 => 9,
                UUID => 16,
                DECIMAL16 => 17,
                BINARY | LONG_STRING => 4 + read_uint(value.get(1..)?, 4)?,
                _ => return None,
            },
        ),
        OBJECT => {
            let offset_size = (bits & 0b11) + 1;
            let id_size = ((bits >> 2) & 0b11) + 1;
            let len_size = if bits & 0b1_0000 != 0 { 4 } else { 1 };
            let len = read_uint(value.get(1..)?, len_size)?;
            let offsets = 1 + len_size + len * id_size;
            let last = offsets + len * offset_size;
            let values = read_uint(value.get(last..)?, offset_size)?;
            Some(last + offset_size + values)
        }
        ARRAY => {
            let offset_size = (bits & 0b11) + 1;
            let len_size = if bits & 0b100 != 0 { 4 } else { 1 };
            let len = read_uint(value.get(1..)?, len_size)?;
            let last = 1 + len_size + len * offset_size;
            let values = read_uint(value.get(last..)?, offset_size)?;
            Some(last + offset_size + values)
        }
        _ => None,
    }
}

/// Writes the fields of `object` that are not `taken` to `out`, as an object of their own.
///
/// The fields keep their ids, so the object reads with the metadata of the row, and their
/// order, which is the order of their names.
fn write_object(object: &Object<'_>, taken: &[bool], out: &mut Vec<u8>) -> Option<()> {
    let mut len = 0;
    let mut size = 0;
    for (position, _) in taken.iter().enumerate().filter(|(_, taken)| !**taken) {
        let value = object.value(position)?;
        len += 1;
        size += value_len(value)?;
    }

    let id_size = object.id_size();
    let offset_size = uint_size(size)?;
    let is_large = len > usize::from(u8::MAX);
    let header = (u8::from(is_large) << 4)
        | (u8::try_from(id_size - 1).ok()? << 2)
        | u8::try_from(offset_size - 1).ok()?;

    out.clear();
    out.push(header << 2 | OBJECT);
    write_uint(out, len, if is_large { 4 } else { 1 });
    for (position, _) in taken.iter().enumerate().filter(|(_, taken)| !**taken) {
        out.extend_from_slice(object.id_bytes(position)?);
    }

    let mut offset = 0;
    for (position, _) in taken.iter().enumerate().filter(|(_, taken)| !**taken) {
        write_uint(out, offset, offset_size);
        offset += value_len(object.value(position)?)?;
    }
    write_uint(out, offset, offset_size);

    for (position, _) in taken.iter().enumerate().filter(|(_, taken)| !**taken) {
        let value = object.value(position)?;
        out.extend_from_slice(value.get(..value_len(value)?)?);
    }
    Some(())
}

/// The number of bytes of the smallest unsigned integer that holds `value`, at most four.
fn uint_size(value: usize) -> Option<usize> {
    Some(match value {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFF_FFFF => 3,
        _ => {
            u32::try_from(value).ok()?;
            4
        }
    })
}

fn write_uint(out: &mut Vec<u8>, value: usize, width: usize) {
    out.extend_from_slice(&value.to_le_bytes()[..width]);
}
