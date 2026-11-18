// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the row encoder.

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::test_harness::ExpectedOutcome;
use vortex_array::test_harness::assert_execution_paths;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_json::Json;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;

use crate::RowEncode;
use crate::RowEncoder;
use crate::RowEncodingOptions;
use crate::RowSize;
use crate::RowSortField;
use crate::compute_row_sizes_with_options;
use crate::convert_columns;
use crate::convert_columns_with_options;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn collect_row_bytes(array: &ListViewArray) -> Vec<Vec<u8>> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        array.dtype(),
        &DType::list(
            DType::Primitive(PType::U8, Nullability::NonNullable),
            Nullability::NonNullable
        )
    );
    let nrows = array.len();
    (0..nrows)
        .map(|i| {
            let slice = array.list_elements_at(i, &mut ctx).unwrap();
            let p = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
            p.as_slice::<u8>().to_vec()
        })
        .collect()
}

fn assert_row_order(rows: &[Vec<u8>], len: usize, compare: impl Fn(usize, usize) -> Ordering) {
    assert_eq!(rows.len(), len);
    for i in 0..len {
        for j in 0..len {
            assert_eq!(rows[i].cmp(&rows[j]), compare(i, j), "rows {i} and {j}");
        }
    }
}

fn assert_sort_order_i64(values: Vec<i64>, descending: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let field = RowSortField::new(descending, true);
    let encoded = convert_columns(&[col], &[field], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, values.len(), |i, j| {
        if descending {
            values[j].cmp(&values[i])
        } else {
            values[i].cmp(&values[j])
        }
    });
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
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u32> = vec![0, 1, 100, u32::MAX, 42, 17];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(&values[j]));
    Ok(())
}

#[rstest]
#[case::ascending_nulls_first(RowSortField::ascending())]
#[case::ascending_nulls_last(RowSortField::ascending().nulls_last())]
#[case::descending_nulls_first(RowSortField::descending())]
#[case::descending_nulls_last(RowSortField::descending().nulls_last())]
fn row_encoding_orders_timestamp_by_value(#[case] field: RowSortField) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = [Some(500i64), None, Some(-100), Some(0), None, Some(500)];
    let storage = PrimitiveArray::from_option_iter(values).into_array();
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let timestamp = ExtensionArray::try_new(dtype, storage.clone())?.into_array();
    let options = RowEncodingOptions::new([field]);
    let encoder = RowEncoder::with_options(options.clone());
    let cols = std::slice::from_ref(&timestamp);

    let encoded = encoder.encode(cols, &mut ctx)?;
    assert_row_order(&collect_row_bytes(&encoded), values.len(), |i, j| {
        match (values[i], values[j]) {
            (Some(a), Some(b)) if field.descending => b.cmp(&a),
            (Some(a), Some(b)) => a.cmp(&b),
            _ if field.nulls_first => values[i].cmp(&values[j]),
            _ => values[j].cmp(&values[i]),
        }
    });

    let storage_cols = std::slice::from_ref(&storage);
    let expected_rows = encoder.encode(storage_cols, &mut ctx)?.into_array();
    let expected_sizes = encoder.row_sizes(storage_cols, &mut ctx)?;
    assert_arrays_eq!(encoded, &expected_rows, &mut ctx);
    assert_arrays_eq!(
        encoder.row_sizes(cols, &mut ctx)?,
        &expected_sizes,
        &mut ctx
    );

    for (function, expected) in [
        (RowEncode.bind(options.clone()), expected_rows),
        (RowSize.bind(options), expected_sizes),
    ] {
        assert_execution_paths(
            cols,
            |inputs| {
                Ok(ScalarFnArray::try_new(function.clone(), inputs.iter().cloned())?.into_array())
            },
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );
    }

    Ok(())
}

#[rstest]
#[case::structure(timestamp_struct_rows())]
#[case::list(timestamp_list_rows())]
#[case::fixed_size_list(timestamp_fixed_list_rows(false))]
#[case::fixed_size_list_of_lists(timestamp_fixed_list_rows(true))]
#[case::map(timestamp_map_rows())]
fn row_encoding_accepts_nested_ordered_extension(
    #[case] column: VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = RowEncoder::default().encode(&[column?], &mut ctx)?;

    // In each fixture rows 1 and 4 are null, then rows 3, 2, and 0 sort in ascending order.
    let value_ranks = [3, 0, 2, 1, 0];
    assert_row_order(&collect_row_bytes(&rows), value_ranks.len(), |i, j| {
        value_ranks[i].cmp(&value_ranks[j])
    });

    Ok(())
}

const NESTED_TIMESTAMP_VALIDITY: [bool; 5] = [true, false, true, true, false];

fn timestamp_millis(values: impl IntoIterator<Item = i64>) -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::from_iter(values).into_array();
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    Ok(ExtensionArray::try_new(dtype, storage)?.into_array())
}

fn timestamp_struct_rows() -> VortexResult<ArrayRef> {
    let timestamps = timestamp_millis([2, 99, 0, -1, -99])?;
    Ok(StructArray::try_new(
        ["ts"].into(),
        vec![timestamps],
        NESTED_TIMESTAMP_VALIDITY.len(),
        Validity::from_iter(NESTED_TIMESTAMP_VALIDITY),
    )?
    .into_array())
}

fn timestamp_list_rows() -> VortexResult<ArrayRef> {
    let timestamps = timestamp_millis([2, 9, 2, -3, -1])?;
    Ok(nullable_timestamp_lists(timestamps)?.into_array())
}

fn timestamp_fixed_list_rows(variable_width: bool) -> VortexResult<ArrayRef> {
    let timestamps = timestamp_millis([2, 9, 99, 99, 2, -3, -1, 7, -99, -99])?;
    let elements = if variable_width {
        // Each timestamp is in a singleton list so the fixed-size list takes its
        // variable-width path while preserving the timestamp order.
        ListViewArray::try_new(
            timestamps,
            PrimitiveArray::from_iter(0..10u32).into_array(),
            buffer![1u32; 10].into_array(),
            Validity::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array()
    } else {
        timestamps
    };
    Ok(FixedSizeListArray::try_new(
        elements,
        2,
        Validity::from_iter(NESTED_TIMESTAMP_VALIDITY),
        NESTED_TIMESTAMP_VALIDITY.len(),
    )?
    .into_array())
}

fn timestamp_map_rows() -> VortexResult<ArrayRef> {
    let keys = buffer![0i32, 1, 0, 1, 0].into_array();
    let values = timestamp_millis([2, 9, 2, -3, -1])?;
    let dtype = MapDType::try_new(keys.dtype().clone(), values.dtype().clone(), true)?;
    let entries = StructArray::from_fields(&[("key", keys), ("value", values)])?.into_array();
    Ok(MapArray::try_new(dtype, nullable_timestamp_lists(entries)?)?.into_array())
}

fn nullable_timestamp_lists(elements: ArrayRef) -> VortexResult<ListViewArray> {
    ListViewArray::try_new(
        elements,
        buffer![0u32, 0, 2, 4, 0].into_array(),
        buffer![2u32, 1, 2, 1, 2].into_array(),
        Validity::from_iter(NESTED_TIMESTAMP_VALIDITY),
        &mut SESSION.create_execution_ctx(),
    )
}

#[rstest]
#[case::json(json_array())]
#[case::vector(Vector::try_new_vector_array(FixedSizeListArray::new(
    buffer![1.0f32, 2.0].into_array(), 2, Validity::NonNullable, 1,
).into_array()))]
#[case::nested_json(StructArray::from_fields(&[
    ("label", VarBinViewArray::from_iter_str(["a"]).into_array()),
    ("doc", json_array()?),
]).map(IntoArray::into_array))]
fn row_encoding_rejects_types_without_ord(#[case] col: VortexResult<ArrayRef>) -> VortexResult<()> {
    let col = col?;
    let mut ctx = SESSION.create_execution_ctx();
    let cols = std::slice::from_ref(&col);
    let encoder = RowEncoder::default();
    let options = RowEncodingOptions::default_for_columns(1);

    for (function, direct_error) in [
        (
            RowEncode.bind(options.clone()),
            encoder.encode(cols, &mut ctx).err(),
        ),
        (
            RowSize.bind(options),
            encoder.row_sizes(cols, &mut ctx).err(),
        ),
    ] {
        let dtype = col.dtype();
        let id = function.id();
        let expected = format!("{id} requires the Ord claim, but {dtype} does not hold it");
        let direct_error = direct_error.expect("the direct encoder must check for Ord");
        let bound_error = ScalarFnArray::try_new(function, [col.clone()])
            .expect_err("binding must check for Ord");
        for error in [direct_error, bound_error] {
            assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
        }
    }

    Ok(())
}

fn json_array() -> VortexResult<ArrayRef> {
    ExtensionArray::try_from_storage::<Json>(
        EmptyMetadata,
        VarBinViewArray::from_iter_str(["{}"]).into_array(),
    )
    .map(IntoArray::into_array)
}

#[test]
fn primitive_f64_sort_order() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // We use IEEE total-ordering semantics: -0.0 < +0.0 in the byte encoding (matches
    // `arrow-row`). Avoid -0.0 in the natural-order baseline since partial_cmp says
    // -0.0 == 0.0.
    let values: Vec<f64> = vec![-1.5, 0.0, 1.5, f64::INFINITY, f64::NEG_INFINITY, PI];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, values.len(), |i, j| {
        values[i].partial_cmp(&values[j]).unwrap()
    });
    Ok(())
}

#[test]
fn bool_sort_order() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let col = BoolArray::from_iter([true, false, true, false]).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let values = [true, false, true, false];
    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(&values[j]));
    Ok(())
}

#[test]
fn utf8_sort_order() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = vec![
        "banana",
        "apple",
        "",
        "cherry",
        "ban",
        "banana_loaf_for_test",
    ];
    let col = VarBinViewArray::from_iter_str(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(values[j]));
    Ok(())
}

#[test]
fn multi_column_sort() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ints: Vec<i32> = vec![1, 2, 1, 2, 1, 3];
    let strs = vec!["b", "a", "a", "b", "c", "z"];
    let col0 = PrimitiveArray::from_iter(ints.clone()).into_array();
    let col1 = VarBinViewArray::from_iter_str(strs.clone()).into_array();
    let encoded = convert_columns(
        &[col0, col1],
        &[RowSortField::default(), RowSortField::default()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, ints.len(), |i, j| {
        ints[i].cmp(&ints[j]).then_with(|| strs[i].cmp(strs[j]))
    });
    Ok(())
}

#[test]
fn nulls_first_and_last() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<Option<i32>> = vec![Some(5), None, Some(1), None, Some(3)];
    let col = PrimitiveArray::from_option_iter(values.clone()).into_array();

    // nulls_first=true
    let encoded = convert_columns(
        std::slice::from_ref(&col),
        &[RowSortField::ascending()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);
    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(&values[j]));
    let mut sorted = rows;
    sorted.sort();
    // The first two sorted entries should be nulls
    let null_count = values.iter().filter(|v| v.is_none()).count();
    for i in 0..null_count {
        // a null encoded row begins with 0x00
        assert_eq!(sorted[i][0], 0x00);
    }
    // nulls_first=false
    let encoded = convert_columns(&[col], &[RowSortField::ascending().nulls_last()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    assert_row_order(&rows, values.len(), |i, j| match (values[i], values[j]) {
        (Some(a), Some(b)) => a.cmp(&b),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
    });
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
fn reusable_options_helpers() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let options = RowEncodingOptions::new([RowSortField::descending().nulls_last()]);
    assert_eq!(options.len(), 1);
    assert!(!options.is_empty());
    assert_eq!(
        options.fields(),
        &[RowSortField {
            descending: true,
            nulls_first: false
        }]
    );

    let col = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let encoder = RowEncoder::with_options(options.clone());
    assert_eq!(encoder.options(), Some(&options));

    let expected_rows = vec![
        vec![1, 127, 255, 255, 254],
        vec![1, 127, 255, 255, 253],
        vec![1, 127, 255, 255, 252],
    ];
    let expected_sizes = StructArray::from_fields(&[
        ("fixed", PrimitiveArray::from_iter([5u32; 3]).into_array()),
        ("var", PrimitiveArray::from_iter([0u32; 3]).into_array()),
    ])?;
    let encoded = encoder.encode(std::slice::from_ref(&col), &mut ctx)?;
    assert_eq!(collect_row_bytes(&encoded), expected_rows);

    let sizes = encoder.row_sizes(std::slice::from_ref(&col), &mut ctx)?;
    assert_arrays_eq!(sizes, &expected_sizes, &mut ctx);

    let encoded = convert_columns_with_options(std::slice::from_ref(&col), &options, &mut ctx)?;
    assert_eq!(collect_row_bytes(&encoded), expected_rows);

    let sizes = compute_row_sizes_with_options(std::slice::from_ref(&col), &options, &mut ctx)?;
    assert_arrays_eq!(sizes, &expected_sizes, &mut ctx);
    Ok(())
}

#[test]
fn row_encoder_new_accepts_sort_fields() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoder = RowEncoder::new([RowSortField::ascending()]);
    let col = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

    let encoded = encoder.encode(std::slice::from_ref(&col), &mut ctx)?;
    assert_eq!(
        collect_row_bytes(&encoded),
        vec![
            vec![1, 128, 0, 0, 1],
            vec![1, 128, 0, 0, 2],
            vec![1, 128, 0, 0, 3]
        ]
    );
    Ok(())
}

#[test]
fn default_row_encoder_uses_default_fields() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let col0 = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let col1 = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();

    let encoded = RowEncoder::default().encode(&[col0, col1], &mut ctx)?;
    assert_eq!(
        collect_row_bytes(&encoded),
        vec![
            vec![1, 128, 0, 0, 1, 1, 128, 0, 0, 4],
            vec![1, 128, 0, 0, 2, 1, 128, 0, 0, 5],
            vec![1, 128, 0, 0, 3, 1, 128, 0, 0, 6],
        ]
    );
    Ok(())
}

#[test]
fn struct_sort_order() -> VortexResult<()> {
    use vortex_array::arrays::StructArray;
    let mut ctx = SESSION.create_execution_ctx();
    let ids: Vec<i64> = vec![3, 1, 3, 1, 2];
    let names = vec!["b", "a", "a", "b", "z"];
    let id_arr = PrimitiveArray::from_iter(ids.clone()).into_array();
    let name_arr = VarBinViewArray::from_iter_str(names.clone()).into_array();
    let struct_arr = StructArray::from_fields(&[("id", id_arr), ("name", name_arr)])?.into_array();

    let encoded = convert_columns(&[struct_arr], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    assert_row_order(&rows, ids.len(), |i, j| {
        ids[i].cmp(&ids[j]).then_with(|| names[i].cmp(names[j]))
    });
    Ok(())
}

#[test]
fn row_size_struct_shape() -> VortexResult<()> {
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;

    use crate::compute_row_sizes;

    let mut ctx = SESSION.create_execution_ctx();
    let ints: Vec<i32> = vec![1, 2, 3, 4, 5];
    let strs = vec!["a", "bb", "ccc", "", "eeeee"];
    let col0 = PrimitiveArray::from_iter(ints).into_array();
    let col1 = VarBinViewArray::from_iter_str(strs).into_array();

    let sizes = compute_row_sizes(
        &[col0, col1],
        &[RowSortField::default(), RowSortField::default()],
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
    let expected = StructArray::from_fields(&[
        ("fixed", PrimitiveArray::from_iter([5u32; 5]).into_array()),
        (
            "var",
            PrimitiveArray::from_iter([34u32, 34, 34, 1, 34]).into_array(),
        ),
    ])?;
    assert_arrays_eq!(struct_arr, expected, &mut ctx);
    Ok(())
}

#[test]
fn single_buffer_invariant() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
        &[RowSortField::default(), RowSortField::default()],
        &mut ctx,
    )?;

    let rows = collect_row_bytes(&encoded);
    assert_eq!(rows.len(), nrows);
    assert!(rows.iter().all(|row| row.len() == 43));
    assert!(encoded.elements().is::<Primitive>());
    assert_eq!(encoded.elements().buffer_handles().len(), 1);
    let expected_total = 64 * 43;

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
    assert_row_order(&rows, nrows, |i, j| i.cmp(&j));
    Ok(())
}

/// Regression: with the previous 2-sentinel varlen scheme, an empty col1 followed by a
/// non-empty col1 that happened to start with `\0` would corrupt multi-column lex order
/// because col2's first byte aligned against col1's pad in the longer row. With the
/// 3-sentinel scheme byte position 0 alone distinguishes empty from non-empty, so column
/// boundaries always align.
#[test]
fn multi_column_varlen_empty_vs_nul_byte_string() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // col1: empty vs single 0-byte. col2: same int for all rows.
    let col1 = VarBinViewArray::from_iter_str(["", "\0", "a", "ab"]).into_array();
    let col2 = PrimitiveArray::from_iter([1i32, 1, 1, 1]).into_array();
    let encoded = convert_columns(
        &[col1, col2],
        &[RowSortField::default(), RowSortField::default()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    let values = ["", "\0", "a", "ab"];
    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(values[j]));
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
    let mut ctx = SESSION.create_execution_ctx();
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
        &[RowSortField::ascending(), RowSortField::ascending()],
        &mut ctx,
    )?;
    let rows = collect_row_bytes(&encoded);

    let values = [None, Some(""), Some("a"), None, Some("")];
    assert_row_order(&rows, values.len(), |i, j| values[i].cmp(&values[j]));
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
    let mut ctx = SESSION.create_execution_ctx();
    let col = VarBinViewArray::from_iter_str(["a", "", "abc"]).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::descending()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);

    let values = ["a", "", "abc"];
    assert_row_order(&rows, values.len(), |i, j| values[j].cmp(values[i]));
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

    let mut ctx = SESSION.create_execution_ctx();
    // Build a nullable struct{name: utf8} where rows 0 and 2 are null but the underlying
    // child has different length data ("short" vs "much longer text data").
    let names =
        VarBinViewArray::from_iter_str(["short", "x", "much longer text data"]).into_array();
    let field_names = FieldNames::from([FieldName::from("name")]);
    let bits = BitBuffer::from_iter([false, true, false]);
    let validity = Validity::from(bits);
    let struct_arr = StructArray::try_new(field_names, vec![names], 3, validity)?.into_array();

    let encoded = convert_columns(&[struct_arr], &[RowSortField::ascending()], &mut ctx)?;
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
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<f32> = vec![-1.5, 0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    assert_row_order(&rows, values.len(), |i, j| {
        values[i].partial_cmp(&values[j]).unwrap()
    });
    Ok(())
}

#[test]
fn primitive_f16_sort_order() -> VortexResult<()> {
    use vortex_array::dtype::half::f16;
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<f16> = vec![
        f16::from_f32(-1.5),
        f16::from_f32(0.0),
        f16::from_f32(1.5),
        f16::INFINITY,
        f16::NEG_INFINITY,
    ];
    let col = PrimitiveArray::from_iter(values.clone()).into_array();
    let encoded = convert_columns(&[col], &[RowSortField::default()], &mut ctx)?;
    let rows = collect_row_bytes(&encoded);
    assert_row_order(&rows, values.len(), |i, j| {
        values[i].partial_cmp(&values[j]).unwrap()
    });
    Ok(())
}

#[rstest]
#[case::ascending(RowSortField::ascending(), vec![4, 2, 1, 0, 5, 3])]
#[case::descending_nulls_last(
    RowSortField::descending().nulls_last(),
    vec![3, 5, 0, 1, 2, 4]
)]
fn variable_list_sort_order(
    #[case] field: RowSortField,
    #[case] expected_indices: Vec<usize>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // [[1, 2], [1], [], [2], null, [1, 3]]
    let elements = PrimitiveArray::from_iter([1i32, 2, 2, 1, 3]).into_array();
    let offsets = buffer![0u32, 0, 0, 2, 0, 3].into_array();
    let sizes = buffer![2u32, 1, 0, 1, 2, 2].into_array();
    let list = ListViewArray::new(
        elements,
        offsets,
        sizes,
        Validity::from_iter([true, true, true, true, false, true]),
        &mut ctx,
    );

    let rows = collect_row_bytes(&convert_columns(&[list.into_array()], &[field], &mut ctx)?);
    let mut actual_indices: Vec<usize> = (0..rows.len()).collect();
    actual_indices.sort_by(|&a, &b| rows[a].cmp(&rows[b]));
    assert_eq!(actual_indices, expected_indices);
    for pair in expected_indices.windows(2) {
        assert!(rows[pair[0]] < rows[pair[1]]);
    }
    Ok(())
}

#[test]
fn variable_list_prefix_order_precedes_following_column() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // The second column deliberately has the opposite order. The list terminator must decide
    // [1] < [1, 2] before comparison can observe that following column.
    let lists = ListViewArray::new(
        PrimitiveArray::from_iter([1i32, 2]).into_array(),
        buffer![0u32, 0].into_array(),
        buffer![1u32, 2].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .into_array();
    let suffix = PrimitiveArray::from_iter([i64::MAX, i64::MIN]).into_array();
    let rows = collect_row_bytes(&convert_columns(
        &[lists, suffix],
        &[RowSortField::ascending(), RowSortField::ascending()],
        &mut ctx,
    )?);

    assert_eq!(rows.len(), 2);
    assert!(rows[0] < rows[1]);
    Ok(())
}

#[test]
fn variable_list_null_element_prefix_precedes_following_column() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // The child null sentinel and ascending list terminator are both zero.
    // The element marker must decide the list prefix before the following column.
    let mut lists = ListViewArray::new(
        PrimitiveArray::from_option_iter([None, Some(1i32)]).into_array(),
        buffer![0u32, 0].into_array(),
        buffer![1u32, 2].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .into_array();
    let suffix = PrimitiveArray::from_iter([i64::MAX, i64::MIN]).into_array();
    for _depth in 0..3 {
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let rows = collect_row_bytes(&convert_columns(
                    &[lists.clone(), suffix.clone()],
                    &[
                        RowSortField::new(descending, nulls_first),
                        RowSortField::ascending(),
                    ],
                    &mut ctx,
                )?);
                assert_eq!(rows.len(), 2);
                assert_eq!(
                    rows[0].cmp(&rows[1]),
                    if descending {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                );
            }
        }
        lists = ListViewArray::new(
            lists,
            buffer![0u32, 1].into_array(),
            buffer![1u32, 1].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )
        .into_array();
    }
    Ok(())
}

#[test]
fn map_sort_order() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let map_dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
        false,
    )?;
    let keys = PrimitiveArray::from_iter([1i32, 2, 2]).into_array();
    let values = PrimitiveArray::from_iter([10i32, 20, 0]).into_array();
    let entries = StructArray::from_fields(&[("key", keys), ("value", values)])?.into_array();
    // [{1: 10, 2: 20}, {1: 10}, {}, {2: 0}, null]
    let entry_lists = ListViewArray::new(
        entries,
        buffer![0u32, 0, 0, 2, 0].into_array(),
        buffer![2u32, 1, 0, 1, 2].into_array(),
        Validity::from_iter([true, true, true, true, false]),
        &mut ctx,
    );
    let maps = MapArray::new(map_dtype, entry_lists).into_array();

    let rows = collect_row_bytes(&convert_columns(
        &[maps],
        &[RowSortField::ascending()],
        &mut ctx,
    )?);
    let mut actual_indices: Vec<usize> = (0..rows.len()).collect();
    actual_indices.sort_by(|&a, &b| rows[a].cmp(&rows[b]));
    assert_eq!(actual_indices, vec![4, 2, 1, 0, 3]);
    for pair in [4, 2, 1, 0, 3].windows(2) {
        assert!(rows[pair[0]] < rows[pair[1]]);
    }
    Ok(())
}

/// Chunks of one decimal column can compress to different physical value widths. The key
/// width must come from the declared dtype, not the chunk's `values_type`, otherwise keys
/// from different chunks are not memcmp-comparable.
#[rstest]
#[case::ascending(false)]
#[case::descending(true)]
fn decimal_keys_comparable_across_chunk_widths(#[case] descending: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(7, 5);
    let field = RowSortField::new(descending, true);

    // One logical DECIMAL(7, 5) column whose chunks compressed to different physical widths.
    let chunk_i8 = DecimalArray::new(buffer![91i8, -5], dtype, Validity::NonNullable).into_array();
    let chunk_i16 =
        DecimalArray::new(buffer![484i16, 300], dtype, Validity::NonNullable).into_array();
    let values: [i32; 4] = [91, -5, 484, 300];

    let keys_i8 = collect_row_bytes(&convert_columns(&[chunk_i8], &[field], &mut ctx)?);
    let keys_i16 = collect_row_bytes(&convert_columns(&[chunk_i16], &[field], &mut ctx)?);
    assert_eq!(keys_i8.len(), 2);
    assert_eq!(keys_i16.len(), 2);
    let keys = [keys_i8, keys_i16].concat();

    for (i, vi) in values.iter().enumerate() {
        for (j, vj) in values.iter().enumerate() {
            let expected = if descending { vj.cmp(vi) } else { vi.cmp(vj) };
            assert_eq!(
                keys[i].cmp(&keys[j]),
                expected,
                "keys for {vi} and {vj} do not compare like the values"
            );
        }
    }
    Ok(())
}

/// A null slot's backing value is unspecified and might not fit the dtype-derived key width;
/// encoding must ignore it rather than report a spurious overflow.
#[test]
fn decimal_null_slot_garbage_does_not_error() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(7, 5);
    let field = RowSortField::new(false, true);

    let chunk = DecimalArray::new(
        buffer![484i64, 10_000_000_000_000, 91],
        dtype,
        Validity::from_iter([true, false, true]),
    )
    .into_array();

    let keys = collect_row_bytes(&convert_columns(&[chunk], &[field], &mut ctx)?);
    assert_eq!(keys.len(), 3);
    assert!(keys[1] < keys[2], "null must sort before non-nulls");
    assert!(keys[2] < keys[0], "91 must sort before 484");
    Ok(())
}

/// A valid value too large for the dtype-derived key width must fail loudly instead of
/// silently encoding a corrupt key.
#[test]
fn decimal_value_not_fitting_key_width_errors() {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(7, 5);
    let field = RowSortField::new(false, true);

    let chunk = DecimalArray::new(buffer![10_000_000_000_000i64], dtype, Validity::NonNullable)
        .into_array();

    let err = convert_columns(&[chunk], &[field], &mut ctx)
        .expect_err("a valid value wider than the key width must be rejected");
    assert!(
        err.to_string().contains("does not fit"),
        "expected a does-not-fit error, got: {err}"
    );
}

#[test]
fn decimal256_keys_preserve_order_and_physical_width_independence() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(76, 4);
    let large = i256::from_parts(0, 1 << 100);
    let values = [
        i256::from_i128(-1),
        large,
        -large,
        i256::ZERO,
        i256::from_i128(1),
    ];
    for descending in [false, true] {
        for nulls_first in [false, true] {
            let field = RowSortField::new(descending, nulls_first);
            let array = DecimalArray::new(
                values.into_iter().chain([i256::MIN]).collect(),
                dtype,
                Validity::from_iter([true, true, true, true, true, false]),
            )
            .into_array();
            let keys = collect_row_bytes(&convert_columns(&[array], &[field], &mut ctx)?);
            assert_eq!(keys.len(), 6);
            assert!(keys.iter().all(|key| key.len() == 33));
            for (i, left) in values.iter().enumerate() {
                for (j, right) in values.iter().enumerate() {
                    let expected = if descending {
                        right.cmp(left)
                    } else {
                        left.cmp(right)
                    };
                    assert_eq!(keys[i].cmp(&keys[j]), expected);
                }
                assert_eq!(keys[5] < keys[i], nulls_first);
            }
            let narrow =
                DecimalArray::new(buffer![-1i8, 0, 1], dtype, Validity::NonNullable).into_array();
            let narrow_keys = collect_row_bytes(&convert_columns(&[narrow], &[field], &mut ctx)?);
            assert_eq!(
                narrow_keys,
                [keys[0].clone(), keys[3].clone(), keys[4].clone()]
            );
        }
    }
    Ok(())
}
