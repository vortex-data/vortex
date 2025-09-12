// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]
use std::iter;
use std::sync::Arc;

use bytes::Bytes;
use itertools::Itertools;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{
    ChunkedArray, ConstantArray, DecimalArray, ListArray, PrimitiveArray, StructArray, VarBinArray,
    VarBinViewArray,
};
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::{ArrayStreamAdapter, ArrayStreamExt};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::{Buffer, ByteBufferMut, buffer};
use vortex_dict::{DictEncoding, DictVTable};
use vortex_dtype::PType::I32;
use vortex_dtype::{DType, DecimalDType, Nullability, PType, StructFields};
use vortex_error::VortexResult;
use vortex_expr::{PackExpr, and, eq, get_item, gt, gt_eq, lit, lt, lt_eq, or, root, select};
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_scalar::Scalar;
use vortex_scan::ScanBuilder;

use crate::{V1_FOOTER_FBS_SIZE, VERSION, VortexFile, VortexOpenOptions, VortexWriteOptions};

#[test]
fn test_eof_values() {
    // this test exists as a reminder to think about whether we should increment the version
    // when we change the footer
    assert_eq!(VERSION, 1);
    assert_eq!(V1_FOOTER_FBS_SIZE, 32);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_read_simple() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let iter = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_iter()
        .unwrap();

    let mut row_count = 0;

    for array in iter {
        let array = array.unwrap();
        row_count += array.len();
    }

    assert_eq!(row_count, 8);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_round_trip_many_types() {
    let strings = VarBinArray::from(vec!["ab", "foo", "bar"]).into_array();

    let numbers = buffer![1u32, 2, 3].into_array();

    let decimal_2 = DecimalArray::new(
        buffer![100i8, 10i8, 2i8],
        DecimalDType::new(2, 1),
        Validity::from_iter([false, true, false]),
    )
    .into_array();

    let decimal_4 = DecimalArray::new(
        buffer![100i16, 10i16, 2i16],
        DecimalDType::new(4, 2),
        Validity::from_iter([false, true, false]),
    )
    .into_array();

    let decimal_9 = DecimalArray::new(
        buffer![100i32, 10i32, 2i32],
        DecimalDType::new(9, 2),
        Validity::from_iter([false, true, false]),
    )
    .into_array();

    let decimal_17 = DecimalArray::new(
        buffer![100i64, 10i64, 20234i64],
        DecimalDType::new(17, 2),
        Validity::from_iter([false, true, false]),
    )
    .into_array();

    let decimal_35 = DecimalArray::new(
        buffer![100i128, 139348340i128, 23943942i128],
        DecimalDType::new(35, 2),
        Validity::from_iter([true, false, false]),
    )
    .into_array();

    let st = StructArray::from_fields(&[
        ("strings", strings),
        ("numbers", numbers),
        ("decimal_2", decimal_2),
        ("decimal_4", decimal_4),
        ("decimal_9", decimal_9),
        ("decimal_17", decimal_17),
        ("decimal_35", decimal_35),
    ])
    .unwrap();
    let mut buf = ByteBufferMut::empty();

    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let chunks: Vec<_> = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_iter()
        .unwrap()
        .try_collect()
        .unwrap();

    let read = ChunkedArray::try_new(chunks, st.dtype().clone()).unwrap();

    assert_eq!(read.len(), 3);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_read_simple_with_spawn() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let lists = ChunkedArray::from_iter([
        ListArray::from_iter_slow::<i16, _>(
            vec![vec![11, 12], vec![21, 22], vec![31, 32], vec![41, 42]],
            Arc::new(I32.into()),
        )
        .unwrap(),
        ListArray::from_iter_slow::<i8, _>(
            vec![vec![51, 52], vec![61, 62], vec![71, 72], vec![81, 82]],
            Arc::new(I32.into()),
        )
        .unwrap(),
    ])
    .into_array();

    let st =
        StructArray::from_fields(&[("strings", strings), ("numbers", numbers), ("lists", lists)])
            .unwrap();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    assert!(!buf.is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_read_projection() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();
    let strings_dtype = strings.dtype().clone();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        Buffer::copy_from(&numbers_expected[..4]).into_array(),
        Buffer::copy_from(&numbers_expected[4..]).into_array(),
    ])
    .into_array();
    let numbers_dtype = numbers.dtype().clone();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();
    let array = file
        .scan()
        .unwrap()
        .with_projection(select(["strings"], root()))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructFields::new(["strings"].into(), vec![strings_dtype]),
            Nullability::NonNullable,
        )
    );

    let actual = array.to_struct().fields()[0]
        .to_varbinview()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(actual, strings_expected);

    let array = file
        .scan()
        .unwrap()
        .with_projection(select(["numbers"], root()))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructFields::new(["numbers"].into(), vec![numbers_dtype]),
            Nullability::NonNullable,
        )
    );

    let primitive_array = array.to_struct().fields()[0].to_primitive();
    let actual = primitive_array.as_slice::<u32>();
    assert_eq!(actual, numbers_expected);
}

#[test]
#[cfg_attr(miri, ignore)]
fn unequal_batches() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "bob"]).into_array(),
        VarBinArray::from(vec!["baz", "ab", "foo", "bar", "baz", "alice"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4, 5].into_array(),
        buffer![6u32, 7, 8, 9, 10].into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let iter = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_iter()
        .unwrap();

    let mut item_count = 0;

    for array in iter {
        let array = array.unwrap();
        item_count += array.len();

        let numbers = array
            .to_struct()
            .field_by_name("numbers")
            .unwrap()
            .to_primitive();
        assert_eq!(numbers.ptype(), PType::U32);
    }
    assert_eq!(item_count, 10);
}

#[test]
#[cfg_attr(miri, ignore)]
fn write_chunked() {
    let strings = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array();
    let string_dtype = strings.dtype().clone();
    let strings_chunked = ChunkedArray::try_new(iter::repeat_n(strings, 4).collect(), string_dtype)
        .unwrap()
        .into_array();
    let numbers = buffer![1u32, 2, 3, 4].into_array();
    let numbers_dtype = numbers.dtype().clone();
    let numbers_chunked =
        ChunkedArray::try_new(iter::repeat_n(numbers, 4).collect(), numbers_dtype)
            .unwrap()
            .into_array();
    let st = StructArray::try_new(
        ["strings", "numbers"].into(),
        vec![strings_chunked, numbers_chunked],
        16,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let st_dtype = st.dtype().clone();

    let chunked_st = ChunkedArray::try_new(iter::repeat_n(st, 3).collect(), st_dtype)
        .unwrap()
        .into_array();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, chunked_st.to_array_iterator())
        .unwrap();

    let iter = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_iter()
        .unwrap();
    let mut array_len: usize = 0;
    for array in iter {
        array_len += array.unwrap().len();
    }
    assert_eq!(array_len, 48);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_empty_varbin_array_roundtrip() {
    let empty = VarBinArray::from(Vec::<&str>::new()).into_array();

    let st = StructArray::from_fields(&[("a", empty)]).unwrap();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let result = file
        .scan()
        .unwrap()
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), st.dtype());
}

#[test]
#[cfg_attr(miri, ignore)]
fn filter_string() {
    let names_orig = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let ages_orig =
        PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]).into_array();
    let st = StructArray::try_new(
        ["name", "age"].into(),
        vec![names_orig, ages_orig],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let result: Vec<_> = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(eq(get_item("name", root()), lit("Joseph")))
        .into_array_iter()
        .unwrap()
        .try_collect()
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0].to_struct().fields()[0].clone();
    assert_eq!(
        names
            .to_varbinview()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = result[0].to_struct().fields()[1].clone();
    assert_eq!(ages.to_primitive().as_slice::<i32>(), vec![25]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn filter_or() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name", "age"].into(),
        vec![names.into_array(), ages.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let result: Vec<_> = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(or(
            eq(get_item("name", root()), lit("Angela")),
            and(
                gt_eq(get_item("age", root()), lit(20)),
                lt_eq(get_item("age", root()), lit(30)),
            ),
        ))
        .into_array_iter()
        .unwrap()
        .try_collect()
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0].to_struct().fields()[0].clone();
    assert_eq!(
        names
            .to_varbinview()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string(), "Angela".to_string()]
    );
    let ages = result[0].to_struct().fields()[1].clone();
    assert_eq!(
        ages.to_primitive()
            .with_iterator(|iter| iter.map(|x| x.cloned()).collect::<Vec<_>>())
            .unwrap(),
        vec![Some(25), None]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn filter_and() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name", "age"].into(),
        vec![names.into_array(), ages.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let result: Vec<_> = VortexOpenOptions::in_memory()
        .open(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(and(
            gt(get_item("age", root()), lit(21)),
            lt_eq(get_item("age", root()), lit(33)),
        ))
        .into_array_iter()
        .unwrap()
        .try_collect()
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0].to_struct().fields()[0].clone();
    assert_eq!(
        names
            .to_varbinview()
            .with_iterator(|iter| iter
                .map(|s| s.map(|st| unsafe { String::from_utf8_unchecked(st.to_vec()) }))
                .collect::<Vec<_>>())
            .unwrap(),
        vec![Some("Joseph".to_string()), None]
    );
    let ages = result[0].to_struct().fields()[1].clone();
    assert_eq!(ages.to_primitive().as_slice::<i32>(), vec![25, 31]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_with_indices_simple() {
    let expected_numbers_split: Vec<Buffer<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(
            expected_numbers_split
                .iter()
                .cloned()
                .map(IntoArray::into_array),
        )
        .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, expected_array.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    // test no indices
    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::<u64>::empty())
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0_u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();
    let actual_kept_numbers_array = actual_kept_array.fields()[0].to_primitive();

    let expected_kept_numbers: Vec<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .collect();
    let actual_kept_numbers = actual_kept_numbers_array.as_slice::<i16>();

    assert_eq!(expected_kept_numbers, actual_kept_numbers);

    // test all indices
    let actual_array = file
        .scan()
        .unwrap()
        .with_row_indices((0u64..500).collect::<Buffer<_>>())
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();
    let actual_numbers_array = actual_array.fields()[0].to_primitive();
    let actual_numbers = actual_numbers_array.as_slice::<i16>();

    assert_eq!(expected_numbers, actual_numbers);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_with_indices_on_two_columns() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        Buffer::copy_from(&numbers_expected[..4]).into_array(),
        Buffer::copy_from(&numbers_expected[4..]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, st.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let kept_indices = [0_u64, 3, 7];
    let array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct()
        .to_struct();

    let strings_actual = array.fields()[0]
        .to_varbinview()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(
        strings_actual,
        kept_indices
            .iter()
            .map(|&x| strings_expected[x as usize])
            .collect::<Vec<_>>()
    );

    let numbers_actual_array = array.fields()[1].to_primitive();
    let numbers_actual = numbers_actual_array.as_slice::<u32>();
    assert_eq!(
        numbers_actual,
        kept_indices
            .iter()
            .map(|&x| numbers_expected[x as usize])
            .collect::<Vec<u32>>()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_with_indices_and_with_row_filter_simple() {
    let expected_numbers_split: Vec<Buffer<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(
            expected_numbers_split
                .iter()
                .cloned()
                .map(IntoArray::into_array),
        )
        .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, expected_array.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices(Buffer::empty())
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    let actual_kept_numbers_array = actual_kept_array.fields()[0].to_primitive();

    let expected_kept_numbers: Buffer<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .filter(|&x| x > 50)
        .collect();
    let actual_kept_numbers = actual_kept_numbers_array.as_slice::<i16>();

    assert_eq!(expected_kept_numbers.as_slice(), actual_kept_numbers);

    // test all indices
    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices((0..500).collect::<Buffer<_>>())
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    let actual_numbers_array = actual_array.fields()[0].to_primitive();
    let actual_numbers = actual_numbers_array.as_slice::<i16>();

    assert_eq!(
        expected_numbers
            .iter()
            .filter(|&&x| x > 50)
            .cloned()
            .collect::<Vec<_>>(),
        actual_numbers
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn filter_string_chunked() {
    let name_chunk1 =
        VarBinViewArray::from_iter_nullable_str([Some("Joseph"), Some("James"), Some("Angela")])
            .into_array();
    let age_chunk1 = PrimitiveArray::from_option_iter([Some(25_i32), Some(31), None]).into_array();
    let name_chunk2 = VarBinViewArray::from_iter_nullable_str([
        Some("Pharrell".to_owned()),
        Some("Khalil".to_owned()),
        Some("Mikhail".to_owned()),
        None,
    ])
    .into_array();
    let age_chunk2 =
        PrimitiveArray::from_option_iter([Some(57_i32), Some(18), None, Some(32)]).into_array();

    let chunk1 = StructArray::from_fields(&[("name", name_chunk1), ("age", age_chunk1)])
        .unwrap()
        .into_array();
    let chunk2 = StructArray::from_fields(&[("name", name_chunk2), ("age", age_chunk2)])
        .unwrap()
        .into_array();
    let dtype = chunk1.dtype().clone();

    let array = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
        .unwrap()
        .into_array();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, array.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(eq(get_item("name", root()), lit("Joseph")))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    assert_eq!(actual_array.len(), 1);
    let names = &actual_array.fields()[0];
    assert_eq!(
        names
            .to_varbinview()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = &actual_array.fields()[1];
    assert_eq!(ages.to_primitive().as_slice::<i32>(), vec![25]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_pruning_with_or() {
    let letter_chunk1 = VarBinViewArray::from_iter_nullable_str([
        Some("A".to_owned()),
        Some("B".to_owned()),
        Some("D".to_owned()),
    ])
    .into_array();
    let number_chunk1 =
        PrimitiveArray::from_option_iter([Some(25_i32), Some(31), None]).into_array();
    let letter_chunk2 = VarBinViewArray::from_iter_nullable_str([
        Some("G".to_owned()),
        Some("I".to_owned()),
        Some("J".to_owned()),
        None,
    ])
    .into_array();
    let number_chunk2 =
        PrimitiveArray::from_option_iter([Some(4_i32), Some(18), None, Some(21)]).into_array();
    let letter_chunk3 = VarBinViewArray::from_iter_nullable_str([
        Some("L".to_owned()),
        None,
        Some("O".to_owned()),
        Some("P".to_owned()),
    ])
    .into_array();
    let number_chunk3 =
        PrimitiveArray::from_option_iter([Some(10_i32), Some(15), None, Some(22)]).into_array();
    let letter_chunk4 = VarBinViewArray::from_iter_nullable_str([
        Some("X".to_owned()),
        Some("Y".to_owned()),
        Some("Z".to_owned()),
    ])
    .into_array();
    let number_chunk4 =
        PrimitiveArray::from_option_iter([Some(66_i32), Some(77), Some(88)]).into_array();

    let chunk1 = StructArray::from_fields(&[("letter", letter_chunk1), ("number", number_chunk1)])
        .unwrap()
        .into_array();
    let chunk2 = StructArray::from_fields(&[("letter", letter_chunk2), ("number", number_chunk2)])
        .unwrap()
        .into_array();
    let chunk3 = StructArray::from_fields(&[("letter", letter_chunk3), ("number", number_chunk3)])
        .unwrap()
        .into_array();
    let chunk4 = StructArray::from_fields(&[("letter", letter_chunk4), ("number", number_chunk4)])
        .unwrap()
        .into_array();
    let dtype = chunk1.dtype().clone();

    let array = ChunkedArray::try_new(vec![chunk1, chunk2, chunk3, chunk4], dtype)
        .unwrap()
        .into_array();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, array.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(or(
            lt_eq(get_item("letter", root()), lit("J")),
            lt(get_item("number", root()), lit(25)),
        ))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    assert_eq!(actual_array.len(), 10);
    let letters = &actual_array.fields()[0];
    assert_eq!(
        letters
            .to_varbinview()
            .with_iterator(|iter| iter
                .map(|opt| opt.map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) }))
                .collect::<Vec<_>>())
            .unwrap(),
        vec![
            Some("A".to_string()),
            Some("B".to_string()),
            Some("D".to_string()),
            Some("G".to_string()),
            Some("I".to_string()),
            Some("J".to_string()),
            None,
            Some("L".to_string()),
            None,
            Some("P".to_string())
        ]
    );
    let numbers = &actual_array.fields()[1];
    assert_eq!(
        (0..numbers.len())
            .map(|index| -> Option<i32> {
                numbers.scalar_at(index).as_primitive().typed_value::<i32>()
            })
            .collect::<Vec<_>>(),
        vec![
            Some(25),
            Some(31),
            None,
            Some(4),
            Some(18),
            None,
            Some(21),
            Some(10),
            Some(15),
            Some(22)
        ]
    );
}

#[test]
fn test_repeated_projection() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let single_column_array = StructArray::from_fields(&[("strings", strings.clone())])
        .unwrap()
        .into_array();

    let expected = StructArray::from_fields(&[("strings", strings.clone()), ("strings", strings)])
        .unwrap()
        .into_array();

    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, single_column_array.to_array_iterator())
        .unwrap();

    let file = VortexOpenOptions::in_memory().open(buf).unwrap();

    let actual = file
        .scan()
        .unwrap()
        .with_projection(select(["strings", "strings"], root()))
        .into_array_iter()
        .unwrap()
        .read_all()
        .unwrap()
        .to_struct();

    assert_eq!(
        (0..actual.len())
            .map(|index| actual.scalar_at(index))
            .collect_vec(),
        (0..expected.len())
            .map(|index| expected.scalar_at(index))
            .collect_vec()
    );
}

fn chunked_file() -> VortexResult<VortexFile> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();

    let mut writer = vec![];
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut writer, array.to_array_iterator())?;
    let buffer: Bytes = writer.into();
    VortexOpenOptions::in_memory().open(buffer)
}

#[test]
fn basic_file_roundtrip() -> VortexResult<()> {
    let vxf = chunked_file()?;
    let result = vxf.scan()?.into_array_iter()?.read_all()?.to_primitive();

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

    Ok(())
}

#[test]
fn file_excluding_dtype() -> VortexResult<()> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();
    let dtype = array.dtype().clone();

    let mut writer = vec![];
    VortexWriteOptions::default()
        .exclude_dtype()
        .blocking::<SingleThreadRuntime>()
        .write(&mut writer, array.to_array_iterator())?;
    let buffer: Bytes = writer.into();

    // Fail to open without DType.
    let vxf = VortexOpenOptions::in_memory().open(buffer.clone());
    assert!(vxf.is_err(), "Opening without DType should fail");

    let vxf = VortexOpenOptions::in_memory()
        .with_dtype(dtype.clone())
        .open(buffer)?;
    assert_eq!(vxf.dtype(), &dtype);
    assert_eq!(vxf.row_count(), 9);

    Ok(())
}

#[test]
fn file_take() -> VortexResult<()> {
    let vxf = chunked_file()?;
    let result = vxf
        .scan()?
        .with_row_indices(buffer![0, 1, 8])
        .into_array_iter()?
        .read_all()?
        .to_primitive();

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 8]);

    Ok(())
}

#[test]
#[should_panic(
    expected = "FileStatsAccumulator temporarily does not support nullable top-level structs"
)]
fn write_nullable_top_level_struct() {
    let ages = PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]);

    let array = StructArray::try_new(
        ["age"].into(),
        vec![ages.into_array()],
        5,
        Validity::AllValid,
    )
    .unwrap()
    .into_array();

    let mut writer = vec![];
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut writer, array.to_array_iterator())
        .unwrap();
}

fn round_trip(
    array: &dyn Array,
    f: impl Fn(ScanBuilder<ArrayRef>) -> VortexResult<ScanBuilder<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let mut writer = vec![];
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut writer, array.to_array_iterator())?;
    let buffer: Bytes = writer.into();

    let vxf = VortexOpenOptions::in_memory()
        .with_dtype(array.dtype().clone())
        .open(buffer)?;

    assert_eq!(vxf.dtype(), array.dtype());
    assert_eq!(vxf.row_count(), array.len() as u64);

    f(vxf.scan()?)?.into_array_iter()?.read_all()
}

#[test]
fn write_nullable_nested_struct() -> VortexResult<()> {
    let nested_dtype = DType::struct_(
        [(
            "nested_field",
            DType::Primitive(PType::F16, Nullability::Nullable),
        )],
        Nullability::Nullable,
    );

    let struct_ = ConstantArray::new(Scalar::null(nested_dtype.clone()), 3).to_array();

    let array = StructArray::try_new(
        ["struct"].into(),
        vec![struct_.into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();

    let result = round_trip(&array, Ok)?.to_struct();

    assert_eq!(result.len(), 3);
    assert_eq!(result.fields().len(), 1);
    assert!(result.all_valid());

    let nested_struct = result.field_by_name("struct")?.to_struct();
    assert_eq!(nested_struct.dtype(), &nested_dtype);
    assert_eq!(nested_struct.len(), 3);
    assert!(nested_struct.all_invalid());

    Ok(())
}

#[test]
fn scan_empty_fields() -> VortexResult<()> {
    let array = (0..10000).collect::<PrimitiveArray>();

    let result = round_trip(array.as_ref(), |scan| {
        Ok(scan.with_projection(PackExpr::try_new_expr(
            Default::default(),
            vec![],
            Nullability::Nullable,
        )?))
    })?;

    assert_eq!(result.len(), array.len());

    Ok(())
}

#[tokio::test]
async fn test_into_tokio_array_stream() -> VortexResult<()> {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let mut buf = ByteBufferMut::empty();
    VortexWriteOptions::default()
        .write(&mut buf, st.to_array_stream())
        .await?;

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let stream = file.scan().unwrap().into_tokio_array_stream()?;
    let array = stream.read_all().await?;

    assert_eq!(array.len(), 8);

    Ok(())
}

#[test]
fn test_array_stream_no_double_dict_encode() -> VortexResult<()> {
    let num_vals = 2048;
    let mut values = Vec::<i64>::with_capacity(num_vals);
    values.extend(iter::repeat_n(0, num_vals / 2));
    values.extend(iter::repeat_n(1, num_vals / 2));

    let array = PrimitiveArray::from_iter(values).into_array();
    let mut buf = Vec::new();
    VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut buf, array.to_array_iterator())?;
    let file = VortexOpenOptions::in_memory().open(buf)?;
    let read_array = file.scan()?.into_array_iter()?.read_all()?;

    let dict = read_array
        .as_opt::<DictVTable>()
        .expect("expected root to be dictionary");
    assert_ne!(
        dict.codes().encoding().id(),
        DictEncoding.id(),
        "dictionary codes should not be dictionary encoded"
    );
    Ok(())
}

#[tokio::test]
async fn test_writer_basic_push() -> VortexResult<()> {
    let strings = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array();
    let numbers = buffer![1u32, 2, 3, 4].into_array();
    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)])?.into_array();
    let dtype = st.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push(st.clone()).await?;
    let summary = writer.finish().await?;

    assert_eq!(summary.row_count(), 4);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 4);
    assert_eq!(result.dtype(), &dtype);

    Ok(())
}

#[tokio::test]
async fn test_writer_multiple_pushes() -> VortexResult<()> {
    let chunk1 =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3].into_array())])?.into_array();
    let chunk2 =
        StructArray::from_fields(&[("numbers", buffer![4u32, 5, 6].into_array())])?.into_array();
    let chunk3 =
        StructArray::from_fields(&[("numbers", buffer![7u32, 8, 9].into_array())])?.into_array();

    let dtype = chunk1.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push(chunk1).await?;
    writer.push(chunk2).await?;
    writer.push(chunk3).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 9);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 9);
    let numbers = result.to_struct().field_by_name("numbers")?.to_primitive();
    assert_eq!(numbers.as_slice::<u32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);

    Ok(())
}

#[tokio::test]
async fn test_writer_push_stream() -> VortexResult<()> {
    let chunk1 =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3].into_array())])?.into_array();
    let chunk2 =
        StructArray::from_fields(&[("numbers", buffer![4u32, 5, 6].into_array())])?.into_array();

    let dtype = chunk1.dtype().clone();

    let stream = futures::stream::iter(vec![Ok(chunk1), Ok(chunk2)]);
    let sendable_stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype.clone(), stream));

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push_stream(sendable_stream).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 6);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 6);
    let numbers = result.to_struct().field_by_name("numbers")?.to_primitive();
    assert_eq!(numbers.as_slice::<u32>(), &[1, 2, 3, 4, 5, 6]);

    Ok(())
}

#[tokio::test]
async fn test_writer_bytes_written() -> VortexResult<()> {
    let array = StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3, 4, 5].into_array())])?
        .into_array();
    let dtype = array.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype);

    assert_eq!(writer.bytes_written(), 0);

    writer.push(array.clone()).await?;
    writer.push(array).await?;

    let bytes_after_push = writer.bytes_written();
    assert!(
        bytes_after_push > 0,
        "Bytes should have been written after pushing twice"
    );

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 10);

    assert!(!buf.is_empty(), "Buffer should contain data");

    Ok(())
}

#[tokio::test]
async fn test_writer_empty_chunks() -> VortexResult<()> {
    let empty = StructArray::from_fields(&[(
        "numbers",
        PrimitiveArray::new::<u32>(buffer![], Validity::NonNullable).into_array(),
    )])?
    .into_array();
    let non_empty =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2].into_array())])?.into_array();

    let dtype = empty.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push(empty.clone()).await?;
    writer.push(non_empty).await?;
    writer.push(empty).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 2);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 2);
    let numbers = result.to_struct().field_by_name("numbers")?.to_primitive();
    assert_eq!(numbers.as_slice::<u32>(), &[1, 2]);

    Ok(())
}

#[tokio::test]
async fn test_writer_mixed_push_and_stream() -> VortexResult<()> {
    let chunk1 =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2].into_array())])?.into_array();
    let chunk2 =
        StructArray::from_fields(&[("numbers", buffer![3u32, 4].into_array())])?.into_array();
    let chunk3 =
        StructArray::from_fields(&[("numbers", buffer![5u32, 6].into_array())])?.into_array();

    let dtype = chunk1.dtype().clone();

    let stream = futures::stream::iter(vec![Ok(chunk2.clone())]);
    let sendable_stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype.clone(), stream));

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push(chunk1).await?;
    writer.push_stream(sendable_stream).await?;
    writer.push(chunk3).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 6);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 6);
    let numbers = result.to_struct().field_by_name("numbers")?.to_primitive();
    assert_eq!(numbers.as_slice::<u32>(), &[1, 2, 3, 4, 5, 6]);

    Ok(())
}

#[tokio::test]
async fn test_writer_with_complex_types() -> VortexResult<()> {
    let strings = VarBinArray::from(vec!["hello", "world", "test"]).into_array();
    let numbers = buffer![100i32, 200, 300].into_array();
    let lists = ListArray::from_iter_slow::<i16, _>(
        vec![vec![1, 2], vec![3, 4, 5], vec![6]],
        Arc::new(I32.into()),
    )?;

    let chunk = StructArray::from_fields(&[
        ("strings", strings),
        ("numbers", numbers),
        ("lists", lists.into_array()),
    ])?
    .into_array();

    let dtype = chunk.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default().writer(&mut buf, dtype.clone());

    writer.push(chunk).await?;
    let footer = writer.finish().await?;

    assert_eq!(footer.row_count(), 3);

    let file = VortexOpenOptions::in_memory().open(buf)?;
    let result = file.scan()?.into_array_iter()?.read_all()?;

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype(), &dtype);

    let strings_field = result.to_struct().field_by_name("strings").cloned()?;
    let strings = strings_field.to_varbinview().with_iterator(|iter| {
        iter.map(|s| s.map(|st| unsafe { String::from_utf8_unchecked(st.to_vec()) }))
            .collect::<Vec<_>>()
    })?;
    assert_eq!(
        strings,
        vec![
            Some("hello".to_string()),
            Some("world".to_string()),
            Some("test".to_string())
        ]
    );

    Ok(())
}

#[tokio::test]
async fn test_writer_with_statistics() -> VortexResult<()> {
    let array = StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3, 4, 5].into_array())])?
        .into_array();

    let mut buf = ByteBufferMut::empty();
    let mut writer = VortexWriteOptions::default()
        .with_file_statistics(PRUNING_STATS.to_vec())
        .writer(&mut buf, array.dtype().clone());

    writer.push(array).await?;
    let summary = writer.finish().await?;

    assert!(summary.footer().statistics().is_some());
    assert_eq!(summary.row_count(), 5);

    Ok(())
}
