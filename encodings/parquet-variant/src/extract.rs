// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extracts a path of object fields from unshredded Parquet Variant storage, reading the
//! [Variant encoding] in place.
//!
//! This covers the extractions a scan runs most: a string or an integer under a few object
//! fields. Reading the bytes in place skips what the general `variant_get` kernel does for every
//! row and every path: building an Arrow array of the storage, a `Variant` over each row, and a
//! `Variant` over each object on the way down.
//!
//! [`extract`] returns `None` for anything it does not cover, such as a value that would need a
//! cast, and the caller falls back to the general kernel. So it never has to decide how a cast
//! behaves, only to recognise the values that need none.
//!
//! [Variant encoding]: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_mask::Mask;

/// The value of a path in one row.
enum Extracted<'a> {
    /// The path does not exist in the row.
    Missing,
    Utf8(&'a str),
    Int(i64),
}

/// The dtypes [`extract`] produces.
#[derive(Clone, Copy)]
enum Target {
    Utf8,
    Int64,
}

/// Extracts `path`, a path of object fields, from each row of unshredded storage as `dtype`.
///
/// `metadata` and `value` are the storage children, and `valid` the rows that hold a value.
/// Returns `None` if this does not cover the extraction: the dtype is not a string or a 64-bit
/// integer, a row holds a value of another type at the path, or the bytes are not what the
/// encoding says they are. A missing path, or a row without a value, is a null.
pub(crate) fn extract(
    metadata: &VarBinViewArray,
    value: &VarBinViewArray,
    valid: &Mask,
    path: &[&str],
    dtype: &DType,
) -> Option<ArrayRef> {
    let target = match dtype {
        DType::Utf8(_) => Target::Utf8,
        DType::Primitive(PType::I64, _) => Target::Int64,
        _ => return None,
    };
    if path.is_empty() || path.len() > MAX_PATH_LEN {
        return None;
    }

    let len = value.len();
    let mut fields = FieldCache::new(path);
    let mut row = |idx: usize| -> Option<Extracted<'_>> {
        if !valid.value(idx) {
            return Some(Extracted::Missing);
        }
        let fields = fields.resolve(metadata, idx)?;
        extract_row(fields, bytes_at(value, idx), path, target)
    };

    let mut validity = BitBufferMut::with_capacity(len);
    match target {
        Target::Utf8 => {
            // The views are built here rather than by a builder, which would check the UTF-8 of
            // every string a second time.
            let mut views = BufferMut::<BinaryView>::with_capacity(len);
            let mut bytes = ByteBufferMut::empty();
            for idx in 0..len {
                let string = match row(idx)? {
                    Extracted::Utf8(string) => Some(string.as_bytes()),
                    Extracted::Missing => None,
                    Extracted::Int(_) => return None,
                };
                validity.append(string.is_some());
                let string = string.unwrap_or_default();
                let offset = u32::try_from(bytes.len()).ok()?;
                if string.len() > BinaryView::MAX_INLINED_SIZE {
                    bytes.extend_from_slice(string);
                }
                views.push(BinaryView::make_view(string, 0, offset));
            }
            // SAFETY: every view is either inlined or spans a string just copied to the only
            // buffer, and each string was checked to be UTF-8 when it was read.
            let array = unsafe {
                VarBinViewArray::new_unchecked(
                    views.freeze(),
                    Arc::from([bytes.freeze()]),
                    DType::Utf8(Nullability::Nullable),
                    Validity::from(validity.freeze()),
                )
            };
            Some(array.into_array())
        }
        Target::Int64 => {
            let mut values = BufferMut::<i64>::with_capacity(len);
            for idx in 0..len {
                let value = match row(idx)? {
                    Extracted::Int(value) => Some(value),
                    Extracted::Missing => None,
                    Extracted::Utf8(_) => return None,
                };
                validity.append(value.is_some());
                values.push(value.unwrap_or_default());
            }
            Some(
                PrimitiveArray::new(values.freeze(), Validity::from(validity.freeze()))
                    .into_array(),
            )
        }
    }
}

/// The bytes of row `idx`, borrowed from `array`.
fn bytes_at(array: &VarBinViewArray, idx: usize) -> &[u8] {
    let view = &array.views()[idx];
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &array.buffer(view.buffer_index as usize).as_slice()[view.as_range()]
    }
}

/// The longest path [`extract`] covers.
const MAX_PATH_LEN: usize = 8;

/// How to find the fields of a path in the objects of rows that share one metadata.
#[derive(Clone, Copy)]
enum Fields<'a> {
    /// The dictionary id of each field name of the path. [`ABSENT`] is the id of a name that
    /// the dictionary does not hold, which no object has.
    Ids([usize; MAX_PATH_LEN]),
    /// The dictionary holds a name of the path more than once, so that the name has no single
    /// id. The fields are found by name.
    Names(Metadata<'a>),
}

/// An id that is not in any dictionary: the ids of an object are at most four bytes wide.
const ABSENT: usize = usize::MAX;

/// The [`Fields`] of a path, for the metadata of the rows seen most recently.
///
/// Few rows have a metadata of their own: the rows of a table share a handful of shapes, and
/// the metadata child is dictionary encoded. Resolving the names of the path against a metadata
/// once turns the lookup in every object from a binary search over names into a scan for an id.
struct FieldCache<'a> {
    path: &'a [&'a str],
    /// Direct mapped on a hash of the metadata's view. Rows with the same metadata under a
    /// dictionary have the same view.
    slots: Vec<Option<(u128, Fields<'a>)>>,
}

impl<'a> FieldCache<'a> {
    const SLOTS: usize = 256;

    fn new(path: &'a [&'a str]) -> Self {
        Self {
            path,
            slots: vec![None; Self::SLOTS],
        }
    }

    /// The fields of the path for row `idx` of `metadata`.
    #[inline]
    fn resolve(&mut self, metadata: &'a VarBinViewArray, idx: usize) -> Option<Fields<'a>> {
        let key = metadata.views()[idx].as_u128();
        // The top byte of the low half of a multiplicative hash, as the table has 256 slots.
        let mixed = (key ^ (key >> 64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let slot = usize::from(mixed.to_le_bytes()[7]);
        if let Some((cached, fields)) = self.slots[slot]
            && cached == key
        {
            return Some(fields);
        }
        let fields = Self::fields(self.path, Metadata::new(bytes_at(metadata, idx))?)?;
        self.slots[slot] = Some((key, fields));
        Some(fields)
    }

    fn fields(path: &[&str], metadata: Metadata<'a>) -> Option<Fields<'a>> {
        let mut ids = [ABSENT; MAX_PATH_LEN];
        for id in 0..metadata.len {
            let name = metadata.name(id)?;
            for (field, found) in path.iter().zip(ids.iter_mut()) {
                if field.as_bytes() == name {
                    if *found != ABSENT {
                        return Some(Fields::Names(metadata));
                    }
                    *found = id;
                }
            }
        }
        Some(Fields::Ids(ids))
    }
}

/// Follows `path` from `value`, and reads what it reaches as `target`.
///
/// Returns `None` for a value that is neither missing nor of the target's type, and for bytes
/// that are cut short.
fn extract_row<'a>(
    fields: Fields<'_>,
    mut value: &'a [u8],
    path: &[&str],
    target: Target,
) -> Option<Extracted<'a>> {
    for (depth, name) in path.iter().enumerate() {
        // A value that is not an object has no fields.
        let Some(object) = Object::new(value)? else {
            return Some(Extracted::Missing);
        };
        let field = match &fields {
            Fields::Ids(ids) => object.field_with_id(ids[depth])?,
            Fields::Names(metadata) => object.field(metadata, name)?,
        };
        match field {
            Some(field) => value = field,
            None => return Some(Extracted::Missing),
        }
    }

    let header = *value.first()?;
    let payload = &value[1..];
    match (header & BASIC_TYPE_MASK, header >> 2, target) {
        (SHORT_STRING, len, Target::Utf8) => utf8(payload.get(..usize::from(len))?),
        (PRIMITIVE, LONG_STRING, Target::Utf8) => {
            let len = read_uint(payload, 4)?;
            utf8(payload.get(4..4 + len)?)
        }
        (PRIMITIVE, INT8, Target::Int64) => Some(Extracted::Int(i64::from(i8::from_le_bytes(
            array(payload)?,
        )))),
        (PRIMITIVE, INT16, Target::Int64) => Some(Extracted::Int(i64::from(i16::from_le_bytes(
            array(payload)?,
        )))),
        (PRIMITIVE, INT32, Target::Int64) => Some(Extracted::Int(i64::from(i32::from_le_bytes(
            array(payload)?,
        )))),
        (PRIMITIVE, INT64, Target::Int64) => {
            Some(Extracted::Int(i64::from_le_bytes(array(payload)?)))
        }
        // Anything else is for the general kernel to cast, a Variant null included.
        _ => None,
    }
}

fn utf8(bytes: &[u8]) -> Option<Extracted<'_>> {
    std::str::from_utf8(bytes).ok().map(Extracted::Utf8)
}

/// The first `N` bytes of `bytes`.
#[inline]
fn array<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    bytes.get(..N)?.try_into().ok()
}

/// Reads a little-endian unsigned integer of `width` bytes, at most four.
#[inline]
fn read_uint(bytes: &[u8], width: usize) -> Option<usize> {
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
const BASIC_TYPE_MASK: u8 = 0b11;
const PRIMITIVE: u8 = 0;
const SHORT_STRING: u8 = 1;
const OBJECT: u8 = 2;

/// Primitive type ids, the six high bits of a primitive value's first byte.
const INT8: u8 = 3;
const INT16: u8 = 4;
const INT32: u8 = 5;
const INT64: u8 = 6;
const LONG_STRING: u8 = 16;

/// The dictionary of field names of a Variant value.
#[derive(Clone, Copy)]
struct Metadata<'a> {
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
struct Object<'a> {
    len: usize,
    id_size: usize,
    offset_size: usize,
    ids: &'a [u8],
    offsets: &'a [u8],
    values: &'a [u8],
}

impl<'a> Object<'a> {
    /// Reads the object header of `value`, or returns `Some(None)` if it is not an object.
    fn new(value: &'a [u8]) -> Option<Option<Self>> {
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

    /// The value of the field whose name has dictionary id `id`, or `Some(None)` if the object
    /// has no such field.
    #[inline]
    fn field_with_id(&self, id: usize) -> Option<Option<&'a [u8]>> {
        let position = if self.id_size == 1 {
            self.ids.iter().position(|&field| usize::from(field) == id)
        } else {
            let mut position = None;
            for (idx, field) in self.ids.chunks_exact(self.id_size).enumerate() {
                if read_uint(field, self.id_size)? == id {
                    position = Some(idx);
                    break;
                }
            }
            position
        };
        let Some(position) = position else {
            return Some(None);
        };
        let offset = read_uint(
            self.offsets.get(position * self.offset_size..)?,
            self.offset_size,
        )?;
        Some(Some(self.values.get(offset..)?))
    }

    /// The value of the field `name`, or `Some(None)` if the object has no such field.
    fn field(&self, metadata: &Metadata<'_>, name: &str) -> Option<Option<&'a [u8]>> {
        let (mut low, mut high) = (0, self.len);
        while low < high {
            let mid = low + (high - low) / 2;
            let id = read_uint(self.ids.get(mid * self.id_size..)?, self.id_size)?;
            match metadata.name(id)?.cmp(name.as_bytes()) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    let offset = read_uint(
                        self.offsets.get(mid * self.offset_size..)?,
                        self.offset_size,
                    )?;
                    return Some(Some(self.values.get(offset..)?));
                }
            }
        }
        Some(None)
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
        let expected = variant_get(
            &arrow_storage,
            GetOptions::new_with_path(PqVariantPath::try_from(path)?)
                .with_as_type(Some(Arc::clone(&field))),
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
        let path = path.split('.').collect::<Vec<_>>();
        let actual = extract(&metadata, &value, &Mask::new_true(rows.len()), &path, dtype);
        Ok((actual, expected))
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::Nullable)
    }

    fn int64() -> DType {
        DType::Primitive(PType::I64, Nullability::Nullable)
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

    /// A value of another type than the one asked for needs a cast, which is the kernel's.
    #[rstest]
    #[case::integer_as_string(r#"{"a":1}"#, utf8())]
    #[case::string_as_integer(r#"{"a":"1"}"#, int64())]
    #[case::null_as_string(r#"{"a":null}"#, utf8())]
    #[case::object_as_string(r#"{"a":{"b":1}}"#, utf8())]
    #[case::double_as_integer(r#"{"a":1.5}"#, int64())]
    fn leaves_casts_to_the_kernel(#[case] row: &str, #[case] dtype: DType) -> VortexResult<()> {
        // The first row is covered, so only the second can turn the extraction down.
        let covered = if matches!(dtype, DType::Utf8(_)) {
            r#"{"a":"x"}"#
        } else {
            r#"{"a":5}"#
        };
        let rows = vec![covered.to_string(), row.to_string()];

        let (actual, _) = extract_both(&rows, "a", &dtype)?;

        assert!(actual.is_none());
        Ok(())
    }

    /// Rows that hold no value are null, whatever their bytes are.
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
