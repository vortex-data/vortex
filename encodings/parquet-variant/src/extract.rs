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

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
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
    if path.is_empty() {
        return None;
    }

    let len = value.len();
    let row = |idx: usize| -> Option<Extracted<'_>> {
        if !valid.value(idx) {
            return Some(Extracted::Missing);
        }
        let metadata = Metadata::new(bytes_at(metadata, idx))?;
        extract_row(&metadata, bytes_at(value, idx), path, target)
    };

    match target {
        Target::Utf8 => {
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                values.push(match row(idx)? {
                    Extracted::Utf8(value) => Some(value),
                    Extracted::Missing => None,
                    Extracted::Int(_) => return None,
                });
            }
            Some(VarBinViewArray::from_iter_nullable_str(values).into_array())
        }
        Target::Int64 => {
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                values.push(match row(idx)? {
                    Extracted::Int(value) => Some(value),
                    Extracted::Missing => None,
                    Extracted::Utf8(_) => return None,
                });
            }
            Some(PrimitiveArray::from_option_iter(values).into_array())
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

/// Follows `path` from `value`, and reads what it reaches as `target`.
///
/// Returns `None` for a value that is neither missing nor of the target's type, and for bytes
/// that are cut short.
fn extract_row<'a>(
    metadata: &Metadata<'_>,
    mut value: &'a [u8],
    path: &[&str],
    target: Target,
) -> Option<Extracted<'a>> {
    for name in path {
        // A value that is not an object has no fields.
        let Some(object) = Object::new(value)? else {
            return Some(Extracted::Missing);
        };
        match object.field(metadata, name)? {
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
fn array<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    bytes.get(..N)?.try_into().ok()
}

/// Reads a little-endian unsigned integer of `width` bytes, at most four.
fn read_uint(bytes: &[u8], width: usize) -> Option<usize> {
    let bytes = bytes.get(..width)?;
    let mut buffer = [0u8; 4];
    buffer[..width].copy_from_slice(bytes);
    usize::try_from(u32::from_le_bytes(buffer)).ok()
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
struct Metadata<'a> {
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
    use std::sync::Arc;
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
    use vortex_array::dtype::Nullability;
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
