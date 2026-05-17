// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::approx_constant,
    clippy::cloned_ref_to_slice_refs,
    clippy::redundant_clone,
    reason = "tests value clarity over micro-optimization"
)]

//! Tests for the row encoder.

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;

use crate::SortField;
use crate::convert_columns;

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
    let field = SortField {
        descending,
        nulls_first: true,
    };
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

    let mut sorted = rows.clone();
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
    let encoded = convert_columns(&[col], &[SortField::default()], &mut ctx)?;
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
fn primitive_f64_sort_order() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // We use IEEE total-ordering semantics: -0.0 < +0.0 in the byte encoding (matches
    // `arrow-row`). Avoid -0.0 in the natural-order baseline since partial_cmp says
    // -0.0 == 0.0.
    let values: Vec<f64> = vec![-1.5, 0.0, 1.5, f64::INFINITY, f64::NEG_INFINITY, 3.14];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[SortField::default()], &mut ctx)?;
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
    let encoded = convert_columns(&[col], &[SortField::default()], &mut ctx)?;
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
    let encoded = convert_columns(&[col], &[SortField::default()], &mut ctx)?;
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
        &[SortField::default(), SortField::default()],
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
        &[col.clone()],
        &[SortField {
            descending: false,
            nulls_first: true,
        }],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted = rows.clone();
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
        &[SortField {
            descending: false,
            nulls_first: false,
        }],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);
    let mut sorted = rows.clone();
    sorted.sort();
    // The last two sorted entries should be nulls
    for i in 0..null_count {
        let pos = sorted.len() - 1 - i;
        assert_eq!(sorted[pos][0], 0x02);
    }
    Ok(())
}

#[test]
fn dict_path_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let raw = VarBinViewArray::from_iter(
        vec![Some("a"), Some("bb"), Some("a"), Some("ccc"), Some("bb")],
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let dict_arr = dict_encode(&raw)?.into_array();

    let canonical_enc = convert_columns(&[raw], &[SortField::default()], &mut ctx)?;
    let dict_enc = convert_columns(&[dict_arr], &[SortField::default()], &mut ctx)?;

    assert_eq!(
        collect_row_bytes(&canonical_enc),
        collect_row_bytes(&dict_enc)
    );
    Ok(())
}

#[test]
fn constant_path_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let nrows = 8usize;
    let const_arr = ConstantArray::new(42i64, nrows).into_array();
    let canonical = PrimitiveArray::from_iter(vec![42i64; nrows]).into_array();

    let from_const = convert_columns(&[const_arr], &[SortField::default()], &mut ctx)?;
    let from_canon = convert_columns(&[canonical], &[SortField::default()], &mut ctx)?;
    assert_eq!(
        collect_row_bytes(&from_const),
        collect_row_bytes(&from_canon)
    );
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

    let encoded = convert_columns(&[struct_arr], &[SortField::default()], &mut ctx)?;
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

    use crate::compute_row_sizes;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let ints: Vec<i32> = vec![1, 2, 3, 4, 5];
    let strs = vec!["a", "bb", "ccc", "", "eeeee"];
    let col0 = PrimitiveArray::from_iter(ints).into_array();
    let col1 = VarBinViewArray::from_iter_str(strs).into_array();

    let sizes = compute_row_sizes(
        &[col0, col1],
        &[SortField::default(), SortField::default()],
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
    // empty string: sentinel(1) + 1 byte; non-empty: sentinel(1) + 33 bytes (single block).
    let expected: Vec<u32> = vec![34, 34, 34, 2, 34];
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
    let col0 = PrimitiveArray::from_iter(primitives.clone()).into_array();
    let col1 = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let encoded = convert_columns(
        &[col0, col1],
        &[SortField::default(), SortField::default()],
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
