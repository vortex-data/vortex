// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowNativeTypeOp;
use arrow_array::DictionaryArray;
use arrow_array::cast::AsArray;
use arrow_array::new_empty_array;
use arrow_array::types::*;
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::AnyCanonical;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrowExportOptions;
use crate::ExecuteArrowExt;
use crate::executor::execute_arrow_naive;
use crate::executor::primitive::canonical_primitive_to_arrow;

/// Matches the encodings [`to_arrow_dictionary`] joins into an Arrow dictionary without decoding:
/// dictionaries, constants, and chunks of them.
struct ArrowDictExportable;

impl Matcher for ArrowDictExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Dict>() || array.is::<Constant>() || array.is::<Chunked>()).then_some(array)
    }
}

pub(super) fn to_arrow_dictionary(
    array: ArrayRef,
    codes_type: &DataType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    match codes_type {
        DataType::Int8 => to_dictionary::<Int8Type>(array, values_field, options, ctx),
        DataType::Int16 => to_dictionary::<Int16Type>(array, values_field, options, ctx),
        DataType::Int32 => to_dictionary::<Int32Type>(array, values_field, options, ctx),
        DataType::Int64 => to_dictionary::<Int64Type>(array, values_field, options, ctx),
        DataType::UInt8 => to_dictionary::<UInt8Type>(array, values_field, options, ctx),
        DataType::UInt16 => to_dictionary::<UInt16Type>(array, values_field, options, ctx),
        DataType::UInt32 => to_dictionary::<UInt32Type>(array, values_field, options, ctx),
        DataType::UInt64 => to_dictionary::<UInt64Type>(array, values_field, options, ctx),
        _ => vortex_bail!("Unsupported dictionary codes type: {:?}", codes_type),
    }
}

/// Export `array` to an Arrow dictionary with keys of type `K`.
///
/// A dictionary, a constant, or a chunked array whose leaves are all dictionaries or constants
/// exports as one join of its leaves: their values are appended, and the codes of each leaf are
/// shifted past the values before it, so nothing is decoded or hashed. A constant contributes one
/// value, or none if it is null. Values are not deduplicated, which Arrow dictionaries permit.
///
/// A single dictionary hands over its codes and values whenever its valid codes fit `K`, even if
/// it has more values than `K` can index.
///
/// Any other array, joined leaves with more values than `K` can index, sparse joined leaves whose
/// values Arrow can hash, and joined values that do not convert to the values field are decoded
/// and hashed into a dictionary by Arrow.
fn to_dictionary<K: ArrowDictionaryKeyType>(
    array: ArrayRef,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    K::Native: NativePType,
{
    let dtype = array.dtype().clone();
    let len = array.len();

    // Execute every leaf before building anything, so an array that cannot be joined costs no more
    // than its decode. The leaves executed so far are kept for that decode. `pending` holds the
    // arrays still to visit in reverse row order, and chunks, including those that execution
    // reveals, are pushed back onto it.
    let mut pending = vec![array];
    let mut leaves = Vec::new();
    let mut num_values = 0usize;
    // Sparse leaves, such as small slices of a large dictionary, decode their rows when Arrow can
    // hash them, rather than export every value of every leaf.
    let max_joined_values = arrow_packs_dictionary(values_field.data_type()).then(|| {
        len.saturating_mul(SPARSE_VALUES_PER_ROW)
            .max(SPARSE_MIN_VALUES)
    });
    while let Some(leaf) = pending.pop() {
        // Keep an existing empty dictionary: canonicalization can discard its codes, and Arrow
        // cannot repack every extension's storage type into a dictionary.
        let leaf = if ArrowDictExportable::matches(&leaf) {
            leaf
        } else {
            leaf.execute_until::<ArrowDictExportable>(ctx)?
        };
        if let Some(chunked) = leaf.as_opt::<Chunked>() {
            let start = pending.len();
            pending.extend(chunked.non_empty_chunks().cloned());
            pending[start..].reverse();
            continue;
        }
        let leaf_values = if let Some(dict) = leaf.as_opt::<Dict>() {
            Some(dict.values().len())
        } else {
            leaf.as_opt::<Constant>()
                .map(|constant| usize::from(!constant.scalar().is_null()))
        };
        leaves.push(leaf);
        // A single dictionary may have more values than `K` can index, and is not sparse
        // relative to itself, so the count only rules out a join once a second leaf arrives.
        // Dictionaries over huge constant values can overflow the count while having few rows.
        let joinable = leaf_values.is_some_and(|leaf_values| {
            let Some(joined) = num_values.checked_add(leaf_values) else {
                return false;
            };
            num_values = joined;
            leaves.len() == 1
                || (keys_index::<K>(num_values)
                    && max_joined_values.is_none_or(|max| num_values <= max))
        });
        if !joinable {
            leaves.extend(pending.into_iter().rev());
            return decode_to_dictionary::<K>(leaves, dtype, values_field, options, ctx);
        }
    }

    if let [leaf] = leaves.as_slice()
        && let Some(dict) = leaf.as_opt::<Dict>()
    {
        if dict.codes().dtype().as_ptype() == K::Native::PTYPE {
            // Codes that already have the key type are handed over without a copy.
            let codes =
                execute_arrow_naive(dict.codes().clone(), Some(&K::DATA_TYPE), options, ctx)?;
            return single_dictionary::<K>(codes, leaves, dtype, values_field, options, ctx);
        }
        if !keys_index::<K>(num_values) {
            // Codes converted by the join below could wrap, so only valid codes that fit `K`
            // are handed over.
            let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
            let key_dtype = DType::Primitive(K::Native::PTYPE, codes.dtype().nullability());
            return match codes
                .into_array()
                .cast(key_dtype)
                .and_then(|codes| codes.execute::<PrimitiveArray>(ctx))
            {
                Ok(codes) => {
                    let codes = canonical_primitive_to_arrow::<K>(codes, ctx)?;
                    single_dictionary::<K>(codes, leaves, dtype, values_field, options, ctx)
                }
                Err(_) => decode_to_dictionary::<K>(leaves, dtype, values_field, options, ctx),
            };
        }
    }

    let mut codes = BufferMut::<K::Native>::with_capacity(len);
    // The validity of joined leaves is concatenated, and a single leaf keeps its own.
    let mut validity = Validity::from(dtype.nullability());
    let mut joined_validity =
        (dtype.is_nullable() && leaves.len() > 1).then(|| BitBufferMut::with_capacity(len));
    let mut values = Vec::with_capacity(leaves.len());
    let mut base = 0usize;
    for leaf in &leaves {
        if let Some(dict) = leaf.as_opt::<Dict>() {
            let leaf_codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(leaf_codes.ptype(), |P| {
                append_codes::<P, K>(leaf_codes.as_slice::<P>(), base, &mut codes)
            });
            let leaf_validity = leaf_codes.as_ref().validity()?;
            match &mut joined_validity {
                Some(joined) => {
                    append_mask(joined, &leaf_validity.execute_mask(leaf_codes.len(), ctx)?);
                }
                None => validity = leaf_validity,
            }
            base += dict.values().len();
            values.push(dict.values().clone());
        } else {
            let constant = leaf.as_::<Constant>();
            let valid = !constant.scalar().is_null();
            let code = if valid {
                K::Native::usize_as(base)
            } else {
                K::Native::ZERO
            };
            codes.push_n(code, constant.len());
            match &mut joined_validity {
                Some(joined) => joined.append_n(valid, constant.len()),
                None if !valid => validity = Validity::AllInvalid,
                None => {}
            }
            if valid {
                base += 1;
                values.push(ConstantArray::new(constant.scalar().clone(), 1).into_array());
            }
        }
    }
    if let Some(joined) = joined_validity {
        validity = Validity::from_bit_buffer(joined.freeze(), Nullability::Nullable);
    }

    let codes =
        canonical_primitive_to_arrow::<K>(PrimitiveArray::new(codes.freeze(), validity), ctx)?;
    let Ok(values) = export_values(values, &dtype, values_field, options, ctx) else {
        return decode_to_dictionary::<K>(leaves, dtype, values_field, options, ctx);
    };
    // SAFETY: `DictionaryArray::try_new` only checks the valid keys. Each valid code of a
    // dictionary was shifted onto the values its leaf contributed, a constant's valid codes point
    // at its value, and `keys_index` bounds the joined values to what `K` can index.
    let dict =
        unsafe { DictionaryArray::<K>::new_unchecked(codes.as_primitive::<K>().clone(), values) };
    Ok(Arc::new(dict))
}

/// Export `leaves`, a single dictionary, with its codes already converted to `codes`.
///
/// Values that do not convert to the values field are decoded and hashed instead, like joined
/// values, since the rows may only use values that do.
fn single_dictionary<K: ArrowDictionaryKeyType>(
    codes: ArrowArrayRef,
    leaves: Vec<ArrayRef>,
    dtype: DType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let values = leaves[0].as_::<Dict>().values().clone();
    let Ok(values) = values.execute_arrow_with_options(Some(values_field), options, ctx) else {
        return decode_to_dictionary::<K>(leaves, dtype, values_field, options, ctx);
    };
    // SAFETY: the valid codes of a Vortex dictionary index its values, and converting them to
    // `K` kept their values.
    let dict =
        unsafe { DictionaryArray::<K>::new_unchecked(codes.as_primitive::<K>().clone(), values) };
    Ok(Arc::new(dict))
}

/// Convert the joined `values` of the leaves to the values field.
fn export_values(
    mut values: Vec<ArrayRef>,
    dtype: &DType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    match values.len() {
        0 => Ok(new_empty_array(values_field.data_type())),
        1 => values
            .swap_remove(0)
            .execute_arrow_with_options(Some(values_field), options, ctx),
        _ => {
            let values = values
                .into_iter()
                .map(|values| values.cast(dtype.clone()))
                .collect::<VortexResult<Vec<_>>>()?;
            ChunkedArray::try_new(values, dtype.clone())?
                .into_array()
                .execute_arrow_with_options(Some(values_field), options, ctx)
        }
    }
}

/// Joined leaves with more than this many values per row decode their rows instead, when Arrow
/// can hash them.
///
/// The join copies every value of every leaf, while decoding gathers and hashes each row once, and
/// hashing a row costs a few value copies. Joined values also stay allocated for as long as the
/// consumer holds the Arrow dictionary.
const SPARSE_VALUES_PER_ROW: usize = 4;

/// Joined leaves with at most this many values always join, however few rows they have: copying
/// them is cheap, and the dictionary keeps the values and the codes of its leaves as they are.
const SPARSE_MIN_VALUES: usize = 1 << 16;

/// Whether Arrow can hash decoded values of `data_type` into a dictionary, as listed by
/// `arrow_cast`'s dictionary packing.
fn arrow_packs_dictionary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(..)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
            | DataType::Struct(_)
    )
}

/// Whether keys of type `K` can index `num_values` values.
fn keys_index<K: ArrowDictionaryKeyType>(num_values: usize) -> bool {
    num_values == 0 || K::Native::from_usize(num_values - 1).is_some()
}

/// Append `codes` shifted by `base`, converted to keys of type `K`.
///
/// Null codes are shifted like the rest instead of being zeroed, which keeps the loop branch-free.
/// Arrow only requires valid keys to index the values, and the export of a single dictionary has
/// always passed its null codes through unchanged. A null code may hold any value, so the shift
/// and the conversion wrap.
fn append_codes<P: IntegerPType, K: ArrowDictionaryKeyType>(
    codes: &[P],
    base: usize,
    out: &mut BufferMut<K::Native>,
) {
    // Truncating `base` and each code to the key width before adding them gives the same key as
    // adding in `usize`, and keeps the loop in the key width.
    let base = K::Native::usize_as(base);
    out.reserve(codes.len());
    for (dst, &code) in out.spare_capacity_mut().iter_mut().zip(codes) {
        dst.write(K::Native::usize_as(code.as_()).add_wrapping(base));
    }
    // SAFETY: the loop above initialized the first `codes.len()` spare elements.
    unsafe { out.set_len(out.len() + codes.len()) };
}

fn append_mask(validity: &mut BitBufferMut, mask: &Mask) {
    match mask.bit_buffer() {
        AllOr::All => validity.append_n(true, mask.len()),
        AllOr::None => validity.append_n(false, mask.len()),
        AllOr::Some(bits) => validity.append_buffer(bits),
    }
}

/// Decode `leaves`, the leaves of an array that cannot be joined, and hash them into a dictionary.
fn decode_to_dictionary<K: ArrowDictionaryKeyType>(
    leaves: Vec<ArrayRef>,
    dtype: DType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = if leaves.len() == 1 {
        leaves.into_iter().next().vortex_expect("one leaf")
    } else {
        // SAFETY: the leaves of a chunked array have its dtype.
        unsafe { ChunkedArray::new_unchecked(leaves, dtype) }
            .into_array()
            .execute_until::<AnyCanonical>(ctx)?
    };
    // Arrow hides this functionality inside the cast module!
    let array = array.execute_arrow_with_options(Some(values_field), options, ctx)?;
    arrow_cast::cast(
        &array,
        &DataType::Dictionary(
            Box::new(K::DATA_TYPE),
            Box::new(values_field.data_type().clone()),
        ),
    )
    .map_err(VortexError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::DictionaryArray as ArrowDictArray;
    use arrow_array::Int32Array;
    use arrow_array::StringArray;
    use arrow_array::UInt8Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::types::UInt8Type;
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::mask::Mask;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrowSessionExt;
    use crate::ExecuteArrowExt;

    #[track_caller]
    fn check_fixed_size_list_dictionary_round_trip(len: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0..i32::try_from(len * 2)?).into_array(),
            2,
            Validity::NonNullable,
            len,
        )?
        .into_array();
        let source = DictArray::try_new(
            PrimitiveArray::from_iter(0..u32::try_from(len)?).into_array(),
            values,
        )?
        .into_array();
        let arrow = SESSION.arrow();
        let values_field = arrow.to_arrow_field("item", source.dtype())?;
        let field = Field::new(
            "dict",
            dict_type(DataType::Int32, values_field.data_type().clone()),
            false,
        );

        let exported = source.clone().execute_arrow(Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());
        let imported = arrow.from_arrow_array(exported, &field)?;
        vortex_array::assert_arrays_eq!(imported, source, &mut ctx);

        Ok(())
    }

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn dict_type(codes: DataType, values: DataType) -> DataType {
        DataType::Dictionary(Box::new(codes), Box::new(values))
    }

    fn execute(
        array: vortex_array::ArrayRef,
        dt: &DataType,
    ) -> VortexResult<arrow_array::ArrayRef> {
        let field = Field::new("", dt.clone(), array.dtype().is_nullable());
        array.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())
    }

    fn dict_basic_input() -> vortex_array::ArrayRef {
        DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )
        .expect("valid dictionary input")
        .into_array()
    }

    fn dict_with_null_codes_input() -> vortex_array::ArrayRef {
        DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![Some(0u8), None, Some(1)]).into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )
        .expect("valid dictionary input with null codes")
        .into_array()
    }

    #[rstest]
    #[case::constant_null(
        ConstantArray::new(Scalar::null(DType::Utf8(Nullable)), 4).into_array(),
        dict_type(DataType::UInt32, DataType::Utf8),
        Arc::new(vec![None::<&str>, None, None, None].into_iter().collect::<ArrowDictArray<UInt32Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::constant_non_null(
        ConstantArray::new(Scalar::from("hello"), 5).into_array(),
        dict_type(DataType::UInt32, DataType::Utf8),
        Arc::new(vec![Some("hello"); 5].into_iter().collect::<ArrowDictArray<UInt32Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_basic(
        dict_basic_input(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_with_null_codes(
        dict_with_null_codes_input(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("b")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_with_null_codes_widened(
        dict_with_null_codes_input(),
        dict_type(DataType::Int32, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("b")].into_iter().collect::<ArrowDictArray<Int32Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::varbinview_fallback(
        [Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<VarBinViewArray>().into_array(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    fn to_arrow_dictionary(
        #[case] input: vortex_array::ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let actual = execute(input, &target_type)?;
        assert_eq!(expected.as_ref(), actual.as_ref());
        Ok(())
    }

    fn utf8_dict(codes: &[u8], values: &[&str]) -> VortexResult<vortex_array::ArrayRef> {
        Ok(DictArray::try_new(
            PrimitiveArray::from_iter(codes.iter().copied()).into_array(),
            VarBinViewArray::from_iter_str(values).into_array(),
        )?
        .into_array())
    }

    #[test]
    fn chunked_dicts_join_without_hashing() -> VortexResult<()> {
        // The first dictionary's values are nullable, and the second's are not, so its values are
        // cast to the chunked dtype before they are joined.
        let nullable_values = ["a", "b"]
            .into_iter()
            .map(Some)
            .collect::<VarBinViewArray>();
        let chunks = vec![
            DictArray::try_new(
                PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
                nullable_values.into_array(),
            )?
            .into_array(),
            ConstantArray::new(Scalar::null(DType::Utf8(Nullable)), 2).into_array(),
            DictArray::try_new(
                PrimitiveArray::from_option_iter([Some(1u8), None]).into_array(),
                VarBinViewArray::from_iter_str(["c", "a"]).into_array(),
            )?
            .into_array(),
        ];
        let chunked = ChunkedArray::try_new(chunks, DType::Utf8(Nullable))?.into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt32, DataType::Utf8))?;

        // Two values per dictionary and none for the null constant: appended, not deduplicated.
        assert_eq!(actual.as_any_dictionary().values().len(), 4);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        let expected = StringArray::from(vec![
            Some("a"),
            Some("b"),
            Some("a"),
            None,
            None,
            Some("a"),
            None,
        ]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }

    /// A chunk of chunks, as a scan concatenating chunked batches produces, joins like its
    /// leaves instead of decoding and hashing its values.
    #[test]
    fn nested_chunked_dicts_join_without_hashing() -> VortexResult<()> {
        let inner = ChunkedArray::try_new(
            vec![
                utf8_dict(&[0, 1], &["a", "b"])?,
                utf8_dict(&[1, 1], &["a", "b"])?,
            ],
            DType::Utf8(NonNullable),
        )?;
        let chunked = ChunkedArray::try_new(
            vec![inner.into_array(), utf8_dict(&[1, 0], &["a", "b"])?],
            DType::Utf8(NonNullable),
        )?
        .into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt32, DataType::Utf8))?;

        // Each leaf's two values are appended, not deduplicated.
        assert_eq!(actual.as_any_dictionary().values().len(), 6);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        let expected = StringArray::from(vec!["a", "b", "b", "b", "b", "a"]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }

    /// Chunks that cannot be joined are decoded and hashed: a chunk that is not a dictionary, and
    /// dictionaries with more values together than the key type can index.
    #[rstest]
    #[case::decoded_chunk(
        DataType::Int32,
        Ok(VarBinViewArray::from_iter_str(["b", "c"]).into_array()),
        vec!["a", "b", "b", "c"],
        3
    )]
    #[case::too_many_values(
        DataType::Int8,
        utf8_dict(&[0, 1, 2], &[["a", "b"].as_slice(), &["x"; 125]].concat()),
        vec!["a", "b", "a", "b", "x"],
        3
    )]
    fn unjoinable_chunks_decode_and_hash(
        #[case] key_type: DataType,
        #[case] second: VortexResult<vortex_array::ArrayRef>,
        #[case] expected: Vec<&str>,
        #[case] distinct: usize,
    ) -> VortexResult<()> {
        let first = utf8_dict(&[0, 1], &["a", "b"])?;
        let dtype = first.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![first, second?], dtype)?.into_array();

        let actual = execute(chunked, &dict_type(key_type, DataType::Utf8))?;

        // Hashing leaves each distinct value once.
        assert_eq!(actual.as_any_dictionary().values().len(), distinct);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        assert_eq!(
            flat.as_ref(),
            &StringArray::from(expected) as &dyn arrow_array::Array
        );
        Ok(())
    }

    /// `Int8` keys index at most 128 values: chunks with exactly that many join, and one more
    /// value decodes and hashes them.
    #[rstest]
    #[case::joins_at_max(124, 128)]
    #[case::hashes_past_max(125, 3)]
    fn int8_keys_join_up_to_max_index(
        #[case] padding: usize,
        #[case] exported_values: usize,
    ) -> VortexResult<()> {
        let first = utf8_dict(&[0, 1], &["a", "b"])?;
        let second = utf8_dict(
            &[0, 1, 2],
            &[["a", "b"].as_slice(), &vec!["x"; padding]].concat(),
        )?;
        let dtype = first.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![first, second], dtype)?.into_array();

        let actual = execute(chunked, &dict_type(DataType::Int8, DataType::Utf8))?;

        assert_eq!(actual.as_any_dictionary().values().len(), exported_values);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        assert_eq!(
            flat.as_ref(),
            &StringArray::from(vec!["a", "b", "a", "b", "x"]) as &dyn arrow_array::Array
        );
        Ok(())
    }

    /// A single dictionary hands over its codes and values even when it has more values than the
    /// key type can index, as long as its codes fit the key type. Arrow cannot hash these values.
    #[rstest]
    #[case::same_ptype(PrimitiveArray::from_iter([0i8, 1, 0]).into_array())]
    #[case::cast_codes(PrimitiveArray::from_option_iter([Some(0u32), None, Some(1)]).into_array())]
    fn single_dict_hands_over_codes_past_key_range(
        #[case] codes: vortex_array::ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0..258i32).into_array(),
            2,
            Validity::NonNullable,
            129,
        )?
        .into_array();
        let source = DictArray::try_new(codes, values)?.into_array();
        let arrow = SESSION.arrow();
        let values_field = arrow.to_arrow_field("item", source.dtype())?;
        let field = Field::new(
            "dict",
            dict_type(DataType::Int8, values_field.data_type().clone()),
            source.dtype().is_nullable(),
        );

        let exported = source.clone().execute_arrow(Some(&field), &mut ctx)?;

        assert_eq!(exported.as_any_dictionary().values().len(), 129);
        let imported = arrow.from_arrow_array(exported, &field)?;
        vortex_array::assert_arrays_eq!(imported, source, &mut ctx);
        Ok(())
    }

    /// A single dictionary whose valid codes do not fit the key type is decoded and hashed.
    #[test]
    fn single_dict_hashes_codes_past_key_range() -> VortexResult<()> {
        let values = (0..201).map(|i| format!("v{i}")).collect::<Vec<_>>();
        let values = values.iter().map(String::as_str).collect::<Vec<_>>();
        let dict = utf8_dict(&[0, 200, 0], &values)?;

        let actual = execute(dict, &dict_type(DataType::Int8, DataType::Utf8))?;

        assert_eq!(actual.as_any_dictionary().values().len(), 2);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        assert_eq!(
            flat.as_ref(),
            &StringArray::from(vec!["v0", "v200", "v0"]) as &dyn arrow_array::Array
        );
        Ok(())
    }

    /// A joined value that no row uses may not convert to the target value type; the rows are
    /// then decoded instead, which only convert the values in use.
    #[test]
    fn unused_value_outside_target_type_decodes_rows() -> VortexResult<()> {
        let chunk = |codes: &[u8], values: &[i8]| -> VortexResult<vortex_array::ArrayRef> {
            Ok(DictArray::try_new(
                PrimitiveArray::from_iter(codes.iter().copied()).into_array(),
                PrimitiveArray::from_iter(values.iter().copied()).into_array(),
            )?
            .into_array())
        };
        let chunked = ChunkedArray::try_new(
            vec![chunk(&[0], &[1, -1])?, chunk(&[0], &[2])?],
            DType::Primitive(PType::I8, NonNullable),
        )?
        .into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt8, DataType::UInt8))?;

        let flat = arrow_cast::cast(&actual, &DataType::UInt8)?;
        assert_eq!(
            flat.as_ref(),
            &UInt8Array::from(vec![1, 2]) as &dyn arrow_array::Array
        );
        Ok(())
    }

    /// A leaf that only becomes chunked when executed, here a mask over chunked dictionaries,
    /// joins its chunks instead of decoding and hashing the whole array.
    #[test]
    fn chunks_revealed_by_execution_join() -> VortexResult<()> {
        let inner = ChunkedArray::try_new(
            vec![
                utf8_dict(&[0, 1], &["a", "b"])?,
                utf8_dict(&[1, 0], &["a", "b"])?,
            ],
            DType::Utf8(NonNullable),
        )?;
        let masked = Mask::try_new(
            inner.into_array(),
            BoolArray::from_iter([true, false, true, true]).into_array(),
        )?
        .into_array();
        let dtype = masked.dtype().clone();
        let tail = utf8_dict(&[1], &["a", "b"])?.cast(dtype.clone())?;
        let chunked = ChunkedArray::try_new(vec![masked, tail], dtype)?.into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt32, DataType::Utf8))?;

        // Each leaf's two values are appended, not deduplicated.
        assert_eq!(actual.as_any_dictionary().values().len(), 6);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        let expected = StringArray::from(vec![Some("a"), None, Some("b"), Some("a"), Some("b")]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }

    /// Slices of a dictionary keep all of its values. Joining sparse slices would export far more
    /// values than rows, so their rows are decoded and hashed instead.
    #[test]
    fn sparse_dict_slices_hash_rows() -> VortexResult<()> {
        let dict = DictArray::try_new(
            PrimitiveArray::from_iter(0..70_000u32).into_array(),
            PrimitiveArray::from_iter((0..70_000i32).map(|i| i % 3)).into_array(),
        )?
        .into_array();
        let chunked = ChunkedArray::try_new(
            vec![dict.slice(0..2)?, dict.slice(69_998..70_000)?],
            dict.dtype().clone(),
        )?
        .into_array();

        let actual = execute(chunked, &dict_type(DataType::Int32, DataType::Int32))?;

        assert_eq!(actual.as_any_dictionary().values().len(), 3);
        let flat = arrow_cast::cast(&actual, &DataType::Int32)?;
        assert_eq!(
            flat.as_ref(),
            &Int32Array::from(vec![0, 1, 2, 0]) as &dyn arrow_array::Array
        );
        Ok(())
    }

    /// Sparse slices still join when Arrow cannot hash their values.
    #[test]
    fn sparse_dict_slices_join_unhashable_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0..140_000i32).into_array(),
            2,
            Validity::NonNullable,
            70_000,
        )?
        .into_array();
        let dict =
            DictArray::try_new(PrimitiveArray::from_iter(0..70_000u32).into_array(), values)?
                .into_array();
        let source = ChunkedArray::try_new(
            vec![dict.slice(0..2)?, dict.slice(69_998..70_000)?],
            dict.dtype().clone(),
        )?
        .into_array();
        let arrow = SESSION.arrow();
        let values_field = arrow.to_arrow_field("item", source.dtype())?;
        let field = Field::new(
            "dict",
            dict_type(DataType::Int32, values_field.data_type().clone()),
            false,
        );

        let exported = source.clone().execute_arrow(Some(&field), &mut ctx)?;

        assert_eq!(exported.as_any_dictionary().values().len(), 140_000);
        let imported = arrow.from_arrow_array(exported, &field)?;
        vortex_array::assert_arrays_eq!(imported, source, &mut ctx);
        Ok(())
    }

    /// Dictionaries over huge constant values can have more values together than `usize` counts,
    /// even when they have few rows. Their rows are decoded and hashed.
    #[test]
    fn value_count_overflow_hashes_rows() -> VortexResult<()> {
        let dict = || -> VortexResult<vortex_array::ArrayRef> {
            Ok(DictArray::try_new(
                PrimitiveArray::from_iter([0u64, 1]).into_array(),
                ConstantArray::new(7i32, usize::MAX / 2 + 1).into_array(),
            )?
            .into_array())
        };
        let chunked = ChunkedArray::try_new(
            vec![dict()?, dict()?],
            DType::Primitive(PType::I32, NonNullable),
        )?
        .into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt64, DataType::Int32))?;

        assert_eq!(actual.as_any_dictionary().values().len(), 1);
        let flat = arrow_cast::cast(&actual, &DataType::Int32)?;
        assert_eq!(
            flat.as_ref(),
            &Int32Array::from(vec![7; 4]) as &dyn arrow_array::Array
        );
        Ok(())
    }

    #[test]
    fn mask_wrapped_dict_exports() -> VortexResult<()> {
        // Dictionary behind a lazy `mask` scalar-fn — the shape a scan produces when a row
        // mask is applied to a dict-encoded column.
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )?;
        let mask = BoolArray::from_iter([true, false, true]);
        let masked = Mask::try_new(dict.into_array(), mask.into_array())?.into_array();

        let actual = execute(masked, &dict_type(DataType::UInt8, DataType::Utf8))?;
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;

        let expected = StringArray::from(vec![Some("a"), None, Some("a")]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }

    #[rstest]
    #[case::empty(0)]
    #[case::nonempty(3)]
    fn fixed_size_list_values_round_trip(#[case] len: usize) -> VortexResult<()> {
        check_fixed_size_list_dictionary_round_trip(len)
    }
}
