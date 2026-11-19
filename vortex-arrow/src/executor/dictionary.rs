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
/// Any other array, and leaves with more values than `K` can index, are decoded and hashed into a
/// dictionary by Arrow.
fn to_dictionary<K: ArrowDictionaryKeyType>(
    array: ArrayRef,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    K::Native: NativePType,
{
    // Keep an existing empty dictionary: canonicalization can discard its codes, and Arrow
    // cannot repack every extension's storage type into a dictionary.
    let array = if array.is::<Dict>() || array.is::<Constant>() {
        array
    } else {
        array.execute_until::<ArrowDictExportable>(ctx)?
    };
    let dtype = array.dtype().clone();
    let len = array.len();

    let mut unexecuted = Vec::new();
    push_leaves(array, &mut unexecuted);

    // Execute every leaf before building anything, so an array that cannot be joined costs no more
    // than its decode. The leaves executed so far are kept for that decode.
    let mut leaves = Vec::with_capacity(unexecuted.len());
    let mut num_values = 0usize;
    let mut unexecuted = unexecuted.into_iter();
    while let Some(leaf) = unexecuted.next() {
        let leaf = if leaf.is::<Dict>() || leaf.is::<Constant>() {
            leaf
        } else {
            leaf.execute_until::<ArrowDictExportable>(ctx)?
        };
        let leaf_values = if let Some(dict) = leaf.as_opt::<Dict>() {
            Some(dict.values().len())
        } else {
            leaf.as_opt::<Constant>()
                .map(|constant| usize::from(!constant.scalar().is_null()))
        };
        leaves.push(leaf);
        let joinable = leaf_values.is_some_and(|leaf_values| {
            num_values += leaf_values;
            keys_index::<K>(num_values)
        });
        if !joinable {
            leaves.extend(unexecuted);
            return decode_to_dictionary::<K>(leaves, dtype, values_field, options, ctx);
        }
    }

    // A single dictionary whose codes already have the key type hands them over without a copy.
    if let [leaf] = leaves.as_slice()
        && let Some(dict) = leaf.as_opt::<Dict>()
        && dict.codes().dtype().as_ptype() == K::Native::PTYPE
    {
        let codes = execute_arrow_naive(dict.codes().clone(), Some(&K::DATA_TYPE), options, ctx)?;
        let values =
            dict.values()
                .clone()
                .execute_arrow_with_options(Some(values_field), options, ctx)?;
        // SAFETY: the valid codes of a Vortex dictionary index its values.
        let dict = unsafe {
            DictionaryArray::<K>::new_unchecked(codes.as_primitive::<K>().clone(), values)
        };
        return Ok(Arc::new(dict));
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
    let values = match values.len() {
        0 => new_empty_array(values_field.data_type()),
        1 => values
            .swap_remove(0)
            .execute_arrow_with_options(Some(values_field), options, ctx)?,
        _ => {
            let values = values
                .into_iter()
                .map(|values| values.cast(dtype.clone()))
                .collect::<VortexResult<Vec<_>>>()?;
            ChunkedArray::try_new(values, dtype)?
                .into_array()
                .execute_arrow_with_options(Some(values_field), options, ctx)?
        }
    };
    // SAFETY: `DictionaryArray::try_new` only checks the valid keys. Each valid code of a
    // dictionary was shifted onto the values its leaf contributed, a constant's valid codes point
    // at its value, and `keys_index` bounds the joined values to what `K` can index.
    let dict =
        unsafe { DictionaryArray::<K>::new_unchecked(codes.as_primitive::<K>().clone(), values) };
    Ok(Arc::new(dict))
}

/// Push the non-empty leaves of `array` in order, looking through chunks of chunks, as from
/// concatenating chunked batches.
fn push_leaves(array: ArrayRef, leaves: &mut Vec<ArrayRef>) {
    if let Some(chunked) = array.as_opt::<Chunked>() {
        for chunk in chunked.non_empty_chunks() {
            push_leaves(chunk.clone(), leaves);
        }
    } else {
        leaves.push(array);
    }
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
    use arrow_array::StringArray;
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
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
