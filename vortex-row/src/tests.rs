// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the row encoder.

use std::f64::consts::PI;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_error::VortexResult;

use crate::RowEncoder;
use crate::RowSortFieldOptions;
use crate::convert_columns;

fn compute_row_sizes(
    cols: &[ArrayRef],
    fields: &[RowSortFieldOptions],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    RowEncoder::new(fields.iter().copied()).row_sizes(cols, ctx)
}

fn collect_row_bytes(array: &ListViewArray) -> Vec<Vec<u8>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let nrows = array.len();
    (0..nrows)
        .map(|i| {
            let slice = array.list_elements_at(i).unwrap();
            let p = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
            p.as_slice::<u8>().to_vec()
        })
        .collect()
}

/// Encode each column independently, sort the resulting row bytes, and check the permutation
/// matches the natural sort order of `values`.
fn assert_sort_order_i64(values: Vec<i64>, descending: bool) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let field = RowSortFieldOptions::new(descending, true);
    let encoded = convert_columns(&[col], &[field], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    // Build expected permutation: sort values naturally then compare to bytes-sorted order.
    let mut idx: Vec<usize> = (0..values.len()).collect();
    if descending {
        idx.sort_by(|a, b| values[*b].cmp(&values[*a]));
    } else {
        idx.sort_by(|a, b| values[*a].cmp(&values[*b]));
    }
    let expected_order: Vec<Vec<u8>> = idx.iter().map(|&i| rows[i].clone()).collect();

    let mut sorted = rows;
    sorted.sort();
    assert_eq!(
        sorted, expected_order,
        "Row-encoded bytes do not match natural sort order"
    );
    Ok(())
}

#[rstest]
#[case::ascending(false)]
#[case::descending(true)]
fn primitive_i64_roundtrip(#[case] descending: bool) -> VortexResult<()> {
    let values: Vec<i64> = vec![-5, 0, 5, i64::MIN, i64::MAX, 7, -7, 1];
    assert_sort_order_i64(values, descending)
}

#[test]
fn primitive_u32_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values: Vec<u32> = vec![0, 1, 100, u32::MAX, 42, 17];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted_rows = rows.clone();
    sorted_rows.sort();

    let mut sorted_idx: Vec<usize> = (0..values.len()).collect();
    sorted_idx.sort_by(|a, b| values[*a].cmp(&values[*b]));
    let expected: Vec<Vec<u8>> = sorted_idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted_rows, expected);
    Ok(())
}

#[test]
fn reject_temporal_extension_dtype_early() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let storage = PrimitiveArray::from_iter([2i32, -1, 0, 7]).into_array();
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
    let col = ExtensionArray::new(ext_dtype, storage).into_array();

    let err = convert_columns(&[col], &[RowSortFieldOptions::ascending()], &mut ctx)
        .expect_err("temporal extensions should be rejected");
    assert!(
        err.to_string().contains("Extension arrays yet"),
        "expected error mentioning unsupported Extension arrays, got: {err}"
    );
    Ok(())
}

#[test]
fn reject_nested_temporal_extension_dtype_early() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let storage = PrimitiveArray::from_iter([2i32, -1, 0, 7]).into_array();
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
    let date_col = ExtensionArray::new(ext_dtype, storage).into_array();
    let tag_col = VarBinViewArray::from_iter_str(["d", "b", "c", "a"]).into_array();
    let struct_col =
        StructArray::from_fields(&[("date", date_col), ("tag", tag_col)])?.into_array();

    let err = convert_columns(&[struct_col], &[RowSortFieldOptions::ascending()], &mut ctx)
        .expect_err("nested temporal extensions should be rejected");
    assert!(
        err.to_string().contains("Extension arrays yet"),
        "expected error mentioning unsupported Extension arrays, got: {err}"
    );
    Ok(())
}

#[test]
fn primitive_f64_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // We use IEEE total-ordering semantics: -0.0 < +0.0 in the byte encoding (matches
    // `arrow-row`). Avoid -0.0 in the natural-order baseline since partial_cmp says
    // -0.0 == 0.0.
    let values: Vec<f64> = vec![-1.5, 0.0, 1.5, f64::INFINITY, f64::NEG_INFINITY, PI];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted_rows = rows.clone();
    sorted_rows.sort();

    let mut sorted_idx: Vec<usize> = (0..values.len()).collect();
    sorted_idx.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap());
    let expected: Vec<Vec<u8>> = sorted_idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted_rows, expected);
    Ok(())
}

#[test]
fn bool_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let col = BoolArray::from_iter([true, false, true, false]).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted = rows.clone();
    sorted.sort();
    // false rows come first (2x), true rows after (2x)
    assert_eq!(sorted[0], rows[1]);
    assert_eq!(sorted[1], rows[3]);
    assert_eq!(sorted[2], rows[0]);
    assert_eq!(sorted[3], rows[2]);
    Ok(())
}

#[test]
fn utf8_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values = vec![
        "banana",
        "apple",
        "",
        "cherry",
        "ban",
        "banana_loaf_for_test",
    ];
    let col = VarBinViewArray::from_iter_str(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted = rows.clone();
    sorted.sort();

    let mut sorted_idx: Vec<usize> = (0..values.len()).collect();
    sorted_idx.sort_by(|a, b| values[*a].cmp(values[*b]));
    let expected: Vec<Vec<u8>> = sorted_idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted, expected);
    Ok(())
}

#[test]
fn multi_column_sort() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let ints: Vec<i32> = vec![1, 2, 1, 2, 1, 3];
    let strs = vec!["b", "a", "a", "b", "c", "z"];
    let col0 = PrimitiveArray::from_iter(ints.clone()).into_array();
    let col1 = VarBinViewArray::from_iter_str(strs.clone()).into_array();
    let encoded = convert_columns(
        &[col0, col1],
        &[
            RowSortFieldOptions::default(),
            RowSortFieldOptions::default(),
        ],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted = rows.clone();
    sorted.sort();
    let mut idx: Vec<usize> = (0..ints.len()).collect();
    idx.sort_by(|a, b| ints[*a].cmp(&ints[*b]).then_with(|| strs[*a].cmp(strs[*b])));
    let expected: Vec<Vec<u8>> = idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted, expected);
    Ok(())
}

#[test]
fn nulls_first_and_last() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values: Vec<Option<i32>> = vec![Some(5), None, Some(1), None, Some(3)];
    let col = PrimitiveArray::from_option_iter(values.clone()).into_array();

    // nulls_first=true
    let encoded = convert_columns(
        std::slice::from_ref(&col),
        &[RowSortFieldOptions::ascending()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted = rows;
    sorted.sort();
    // The first two sorted entries should be nulls
    let null_count = values.iter().filter(|v| v.is_none()).count();
    for i in 0..null_count {
        // a null encoded row begins with 0x00
        assert_eq!(sorted[i][0], 0x00);
    }
    // nulls_first=false
    let encoded = convert_columns(
        &[col],
        &[RowSortFieldOptions::ascending().nulls_last()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted = rows;
    sorted.sort();
    // The last two sorted entries should be nulls
    for i in 0..null_count {
        let pos = sorted.len() - 1 - i;
        assert_eq!(sorted[pos][0], 0x02);
    }
    Ok(())
}

#[test]
fn row_encoder_new_accepts_sort_fields() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoder = RowEncoder::new([RowSortFieldOptions::ascending()]);
    let col = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

    let encoded = encoder.encode(std::slice::from_ref(&col), &mut ctx)?;
    assert_eq!(encoded.len(), 3);
    Ok(())
}

#[test]
fn default_row_encoder_uses_default_fields() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let col0 = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let col1 = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();

    let encoded = RowEncoder::default().encode(&[col0, col1], &mut ctx)?;
    assert_eq!(encoded.len(), 3);
    Ok(())
}

#[test]
fn struct_sort_order() -> VortexResult<()> {
    use vortex_array::arrays::StructArray;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let ids: Vec<i64> = vec![3, 1, 3, 1, 2];
    let names = vec!["b", "a", "a", "b", "z"];
    let id_arr = PrimitiveArray::from_iter(ids.clone()).into_array();
    let name_arr = VarBinViewArray::from_iter_str(names.clone()).into_array();
    let struct_arr = StructArray::from_fields(&[("id", id_arr), ("name", name_arr)])?.into_array();

    let encoded = convert_columns(&[struct_arr], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let mut sorted = rows.clone();
    sorted.sort();
    let mut idx: Vec<usize> = (0..ids.len()).collect();
    idx.sort_by(|a, b| ids[*a].cmp(&ids[*b]).then_with(|| names[*a].cmp(names[*b])));
    let expected: Vec<Vec<u8>> = idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted, expected);
    Ok(())
}

#[test]
fn row_size_struct_shape() -> VortexResult<()> {
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let ints: Vec<i32> = vec![1, 2, 3, 4, 5];
    let strs = vec!["a", "bb", "ccc", "", "eeeee"];
    let col0 = PrimitiveArray::from_iter(ints).into_array();
    let col1 = VarBinViewArray::from_iter_str(strs).into_array();

    let sizes = compute_row_sizes(
        &[col0, col1],
        &[
            RowSortFieldOptions::default(),
            RowSortFieldOptions::default(),
        ],
        &mut ctx,
    )?;
    // Shape must be Struct { fixed, var }
    let struct_arr = sizes.execute::<StructArray>(&mut ctx)?;
    assert_eq!(struct_arr.struct_fields().nfields(), 2);
    let fixed = struct_arr.unmasked_field(0);
    let var = struct_arr.unmasked_field(1);

    // `fixed` must be ConstantArray with value = encoded i32 width = 1 + 4 = 5.
    let fixed_const = fixed
        .as_opt::<Constant>()
        .expect("fixed field should be a ConstantArray");
    assert_eq!(
        fixed_const.scalar(),
        &vortex_array::scalar::Scalar::from(5u32),
        "fixed scalar should be encoded primitive i32 width"
    );

    // `var` must be a PrimitiveArray<u32>, since we have a varlen column.
    let var_prim = var.clone().execute::<PrimitiveArray>(&mut ctx)?;
    let v: &[u32] = var_prim.as_slice();
    assert_eq!(v.len(), 5);
    // empty string: just the empty sentinel (1 byte); null or non-empty:
    // sentinel(1) + 33 bytes (single block).
    let expected: Vec<u32> = vec![34, 34, 34, 1, 34];
    assert_eq!(v, expected.as_slice());
    Ok(())
}

#[test]
fn single_buffer_invariant() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // Encoded rows here are all > 12 bytes, forcing the Ref-view path that points back into
    // the shared data buffer.
    let nrows = 64usize;
    let primitives: Vec<i64> = (0..nrows as i64).collect();
    let strings: Vec<String> = (0..nrows)
        .map(|i| format!("row_{}_with_padding", i))
        .collect();
    let col0 = PrimitiveArray::from_iter(primitives).into_array();
    let col1 = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let encoded = convert_columns(
        &[col0, col1],
        &[
            RowSortFieldOptions::default(),
            RowSortFieldOptions::default(),
        ],
        &mut ctx,
    )?;

    let rows = collect_row_bytes(&encoded);
    let expected_total: usize = rows.iter().map(|r| r.len()).sum();

    // The shared data buffer holds the contiguous concatenation of every row's encoded bytes;
    // per-row allocations would produce many small buffers instead of one shared buffer.
    // ListView's elements array is a single contiguous primitive (u8) array; its length
    // equals the sum of all per-row sizes. A per-row allocation strategy would instead
    // produce N separate elements arrays or a sparse one.
    let elements_len = encoded.elements().len();
    assert_eq!(
        elements_len, expected_total,
        "elements buffer size mismatch"
    );
    Ok(())
}

/// Regression: with the previous 2-sentinel varlen scheme, an empty col1 followed by a
/// non-empty col1 that happened to start with `\0` would corrupt multi-column lex order
/// because col2's first byte aligned against col1's pad in the longer row. With the
/// 3-sentinel scheme byte position 0 alone distinguishes empty from non-empty, so column
/// boundaries always align.
#[test]
fn multi_column_varlen_empty_vs_nul_byte_string() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // col1: empty vs single 0-byte. col2: same int for all rows.
    let col1 = VarBinViewArray::from_iter_str(["", "\0", "a", "ab"]).into_array();
    let col2 = PrimitiveArray::from_iter([1i32, 1, 1, 1]).into_array();
    let encoded = convert_columns(
        &[col1, col2],
        &[
            RowSortFieldOptions::default(),
            RowSortFieldOptions::default(),
        ],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    // Logical natural order of col1: "" < "\0" < "a" < "ab".
    // Byte sort of the encoded rows must put them in that same order.
    let sorted_indices_by_bytes = {
        let mut indices: Vec<usize> = (0..rows.len()).collect();
        indices.sort_by(|a, b| rows[*a].cmp(&rows[*b]));
        indices
    };
    assert_eq!(
        sorted_indices_by_bytes,
        vec![0, 1, 2, 3],
        "byte sort must match natural col1 order; sorted indices were {:?}",
        sorted_indices_by_bytes
    );
    Ok(())
}

/// Regression: null col1 must sort distinct from empty col1 even when col2 follows. With
/// the 3-sentinel scheme null=0x00, empty=0x01 differ at byte 0.
#[test]
fn multi_column_varlen_null_vs_empty() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let col1 = VarBinViewArray::from_iter_nullable_str([
        None::<&str>,
        Some(""),
        Some("a"),
        None,
        Some(""),
    ])
    .into_array();
    let col2 = PrimitiveArray::from_iter([1i32, 1, 1, 1, 1]).into_array();
    let encoded = convert_columns(
        &[col1, col2],
        &[
            RowSortFieldOptions::ascending(),
            RowSortFieldOptions::ascending(),
        ],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    // Nulls first, then empties, then non-empties — and all the col2 values are identical
    // so col1 fully determines the order.
    // Categorise each row by the leading byte of col1's encoding.
    let mut buckets: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, row) in rows.iter().enumerate() {
        let bucket = match row[0] {
            0x00 => 0, // null
            0x01 => 1, // empty
            0x02 => 2, // non-empty
            other => panic!("unexpected varlen sentinel: {:#x}", other),
        };
        buckets[bucket].push(i);
    }
    assert_eq!(buckets[0].len(), 2, "two null col1 rows");
    assert_eq!(buckets[1].len(), 2, "two empty col1 rows");
    assert_eq!(buckets[2].len(), 1, "one non-empty col1 row");

    // All null rows must be byte-equal (same col2 value, both col1 null, single sentinel).
    let null_rows: Vec<&Vec<u8>> = buckets[0].iter().map(|&i| &rows[i]).collect();
    assert_eq!(
        null_rows[0], null_rows[1],
        "null col1 rows must be byte-equal"
    );
    // Same for empty.
    let empty_rows: Vec<&Vec<u8>> = buckets[1].iter().map(|&i| &rows[i]).collect();
    assert_eq!(
        empty_rows[0], empty_rows[1],
        "empty col1 rows must be byte-equal"
    );

    // Byte sort must group: nulls, empties, non-empties (because leading byte differs).
    let mut sorted = rows.clone();
    sorted.sort();
    assert_eq!(sorted[0][0], 0x00);
    assert_eq!(sorted[1][0], 0x00);
    assert_eq!(sorted[2][0], 0x01);
    assert_eq!(sorted[3][0], 0x01);
    assert_eq!(sorted[4][0], 0x02);
    Ok(())
}

/// Regression: descending varlen must put non-empty before empty (natural "" < "a" inverts
/// to "a" < "" under descending). The 3-sentinel scheme uses `!empty < !non_empty` so
/// non-empty's first byte is smaller than empty's first byte.
#[test]
fn varlen_descending_empty_vs_non_empty() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let col = VarBinViewArray::from_iter_str(["a", "", "abc"]).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::descending()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    // Natural order: "" < "a" < "abc"; descending byte sort: "abc" first, "" last.
    let mut sorted = rows.clone();
    sorted.sort();
    // sorted[0] = encoded("abc"), sorted[1] = encoded("a"), sorted[2] = encoded("")
    assert_eq!(sorted[0], rows[2], "abc first in descending");
    assert_eq!(sorted[1], rows[0], "a second");
    assert_eq!(sorted[2], rows[1], "empty last");
    Ok(())
}

/// Regression: two null parent struct rows whose underlying child values differ in length
/// must still produce byte-equal encodings, because the parent emits a canonical null
/// body (one null sentinel per variable child) regardless of the underlying values.
#[test]
fn null_struct_rows_with_varying_child_lengths_are_byte_equal() -> VortexResult<()> {
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // Build a nullable struct{name: utf8} where rows 0 and 2 are null but the underlying
    // child has different length data ("short" vs "much longer text data").
    let names =
        VarBinViewArray::from_iter_str(["short", "x", "much longer text data"]).into_array();
    let field_names = FieldNames::from([FieldName::from("name")]);
    let bits = BitBuffer::from_iter([false, true, false]);
    let validity = Validity::from(bits);
    let struct_arr = StructArray::try_new(field_names, vec![names], 3, validity)?.into_array();

    let encoded = convert_columns(&[struct_arr], &[RowSortFieldOptions::ascending()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    assert_eq!(rows.len(), 3);
    // Both null parent rows must produce identical bytes despite the divergent children.
    assert_eq!(
        rows[0], rows[2],
        "two null parent struct rows must encode to byte-equal slices"
    );
    // And the non-null row's leading sentinel must differ from the null sentinel.
    assert_ne!(rows[0][0], rows[1][0], "null vs non-null sentinel differs");
    Ok(())
}

#[test]
fn primitive_f32_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values: Vec<f32> = vec![-1.5, 0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted_rows = rows.clone();
    sorted_rows.sort();
    let mut sorted_idx: Vec<usize> = (0..values.len()).collect();
    sorted_idx.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap());
    let expected: Vec<Vec<u8>> = sorted_idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted_rows, expected);
    Ok(())
}

#[test]
fn primitive_f16_sort_order() -> VortexResult<()> {
    use vortex_array::dtype::half::f16;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let values: Vec<f16> = vec![
        f16::from_f32(-1.5),
        f16::from_f32(0.0),
        f16::from_f32(1.5),
        f16::INFINITY,
        f16::NEG_INFINITY,
    ];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortFieldOptions::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted_rows = rows.clone();
    sorted_rows.sort();
    let mut sorted_idx: Vec<usize> = (0..values.len()).collect();
    sorted_idx.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap());
    let expected: Vec<Vec<u8>> = sorted_idx.iter().map(|&i| rows[i].clone()).collect();
    assert_eq!(sorted_rows, expected);
    Ok(())
}

#[test]
fn decimal_nullable_sort_order() -> VortexResult<()> {
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // precision=9 -> minimal physical type I32; row 1 is null.
    let dt = DecimalDType::new(9, 3);
    let values: Vec<i32> = vec![5, 0, -7, 0, 123];
    let validity = Validity::from(BitBuffer::from_iter([true, false, true, false, true]));
    let col =
        DecimalArray::new::<i32>(Buffer::<i32>::copy_from(&values), dt, validity).into_array();

    let encoded = convert_columns(&[col], &[RowSortFieldOptions::ascending()], &mut ctx)?;
    let mut sorted = collect_row_bytes(&encoded);
    sorted.sort();
    // nulls_first: the two null rows sort to the front and are byte-equal.
    assert_eq!(sorted[0][0], 0x00, "null sentinel sorts first");
    assert_eq!(sorted[0], sorted[1], "null decimal rows are byte-equal");
    assert_eq!(sorted[1][0], 0x00);
    assert_eq!(sorted[2][0], 0x01, "non-null sentinel");
    Ok(())
}

/// Regression: a decimal column whose physical `values_type` is wider than its precision
/// requires (precision 5 fits in `I32` but is stored as `i64`) must still encode correctly.
/// The size pass reserves the precision-minimal width, so the encode pass must narrow the
/// physical values to that same width rather than writing the wider physical bytes (which
/// previously overran the per-row slot). Byte order must still match the natural value order.
#[rstest]
#[case::ascending(false)]
#[case::descending(true)]
fn decimal_wide_physical_storage_sort_order(#[case] descending: bool) -> VortexResult<()> {
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // precision=5 fits in I32 (4 bytes), but store physically as i64 (8 bytes).
    let dt = DecimalDType::new(5, 2);
    let values: Vec<i64> = vec![1, -4, 0, 99_999, -99_999, 42, -42];
    let col =
        DecimalArray::new::<i64>(Buffer::<i64>::copy_from(&values), dt, Validity::NonNullable)
            .into_array();
    let field = RowSortFieldOptions::new(descending, true);
    let encoded = convert_columns(&[col], &[field], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    // Each encoded row is the precision-minimal width: sentinel(1) + I32(4) = 5 bytes.
    assert!(rows.iter().all(|r| r.len() == 5), "row lens: {:?}", rows);

    let mut idx: Vec<usize> = (0..values.len()).collect();
    if descending {
        idx.sort_by(|a, b| values[*b].cmp(&values[*a]));
    } else {
        idx.sort_by(|a, b| values[*a].cmp(&values[*b]));
    }
    let expected: Vec<Vec<u8>> = idx.iter().map(|&i| rows[i].clone()).collect();
    let mut sorted = rows;
    sorted.sort();
    assert_eq!(
        sorted, expected,
        "decimal byte order must match value order"
    );
    Ok(())
}

/// Regression: a nullable decimal stored wider than its precision requires may hold arbitrary
/// garbage in null slots (`DecimalArray` only constrains *non-null* values to the precision).
/// The narrowing pass must skip null slots instead of panicking when the garbage does not fit
/// the precision-minimal type.
#[test]
fn decimal_wide_storage_with_garbage_null_slot() -> VortexResult<()> {
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // precision=5 -> minimal physical type I32, but stored as i64. Row 1 is null and its slot
    // holds a value that fits neither i32 nor precision 5.
    let dt = DecimalDType::new(5, 2);
    let values: Vec<i64> = vec![7, i64::MAX, -99_999];
    let validity = Validity::from(BitBuffer::from_iter([true, false, true]));
    let col =
        DecimalArray::new::<i64>(Buffer::<i64>::copy_from(&values), dt, validity).into_array();

    let encoded = convert_columns(&[col], &[RowSortFieldOptions::ascending()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    // sentinel(1) + i32(4) per row.
    assert!(rows.iter().all(|r| r.len() == 5), "row lens: {rows:?}");
    // The null row encodes as the canonical null (sentinel 0x00, zero body), so it sorts first
    // and carries no trace of the garbage slot value.
    assert_eq!(rows[1], vec![0x00, 0, 0, 0, 0]);
    let mut sorted = rows.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![rows[1].clone(), rows[2].clone(), rows[0].clone()]
    );
    Ok(())
}

/// Lock-in reference test: encode the worked-example row from `docs/specs/row-encoding.md`
/// (one row with every supported encoding family, all columns ascending nulls-first) and
/// assert the exact encoded bytes. This pins the byte layout so any accidental change to the
/// format is caught, and keeps the spec document honest.
#[test]
fn reference_row_bytes_match_spec() -> VortexResult<()> {
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::NullArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    let null_col = NullArray::new(1).into_array();
    let bool_col = BoolArray::from_iter([true]).into_array();
    let uint_col = PrimitiveArray::from_iter([258u16]).into_array();
    let int_col = PrimitiveArray::from_iter([-5i16]).into_array();
    let float_col = PrimitiveArray::from_iter([1.5f32]).into_array();
    let decimal_col = DecimalArray::new::<i32>(
        Buffer::<i32>::copy_from([12345i32]),
        DecimalDType::new(9, 2),
        Validity::NonNullable,
    )
    .into_array();
    let utf8_col = VarBinViewArray::from_iter_str(["a"]).into_array();
    let binary_col = VarBinViewArray::from_iter_bin([[0xDEu8, 0xAD, 0xBE, 0xEF]]).into_array();
    let struct_col = StructArray::from_fields(&[
        ("x", PrimitiveArray::from_iter([1i8]).into_array()),
        ("y", VarBinViewArray::from_iter_str([""]).into_array()),
    ])?
    .into_array();
    let fsl_col = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter([1u8, 2, 3]).into_array(),
        3,
        Validity::NonNullable,
        1,
    )?
    .into_array();

    let cols = [
        null_col,
        bool_col,
        uint_col,
        int_col,
        float_col,
        decimal_col,
        utf8_col,
        binary_col,
        struct_col,
        fsl_col,
    ];
    let fields = vec![RowSortFieldOptions::default(); cols.len()];
    let encoded = convert_columns(&cols, &fields, &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    assert_eq!(rows.len(), 1);

    // Per-column encodings from the spec's worked example.
    let mut expected: Vec<u8> = Vec::new();
    expected.extend_from_slice(&[0x00]); // null_col
    expected.extend_from_slice(&[0x01, 0x02]); // bool_col: true
    expected.extend_from_slice(&[0x01, 0x01, 0x02]); // uint_col: 258 u16
    expected.extend_from_slice(&[0x01, 0x7F, 0xFB]); // int_col: -5 i16 (sign-bit flipped)
    expected.extend_from_slice(&[0x01, 0xBF, 0xC0, 0x00, 0x00]); // float_col: 1.5 f32
    expected.extend_from_slice(&[0x01, 0x80, 0x00, 0x30, 0x39]); // decimal_col: 12345 i32
    // utf8 "a": non-empty sentinel, 'a', zero pad to 32, length marker 1.
    expected.push(0x02);
    expected.push(b'a');
    expected.extend(std::iter::repeat_n(0u8, 31));
    expected.push(0x01);
    // binary DE AD BE EF: non-empty sentinel, data, zero pad to 32, length marker 4.
    expected.push(0x02);
    expected.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    expected.extend(std::iter::repeat_n(0u8, 28));
    expected.push(0x04);
    // struct { x: 1 i8, y: "" }: outer sentinel, x = 0x01 || 0x81, y = empty sentinel 0x01.
    expected.extend_from_slice(&[0x01, 0x01, 0x81, 0x01]);
    // fsl [1, 2, 3] u8: outer sentinel, then per element 0x01 || BE(value).
    expected.extend_from_slice(&[0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x03]);

    assert_eq!(
        rows[0], expected,
        "encoded reference row does not match the documented byte layout"
    );
    Ok(())
}

#[test]
fn reject_list_dtype_early() {
    use vortex_array::ArrayRef;
    use vortex_array::arrays::ListArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let offsets = PrimitiveArray::new(buffer![0u32, 1, 2], Validity::NonNullable).into_array();
    let elements = PrimitiveArray::from_iter([10i32, 20]).into_array();
    let list: ArrayRef = ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array();
    let err = convert_columns(&[list], &[RowSortFieldOptions::default()], &mut ctx)
        .expect_err("List should not be accepted");
    assert!(
        err.to_string().contains("List"),
        "expected error mentioning List, got: {err}"
    );
}
