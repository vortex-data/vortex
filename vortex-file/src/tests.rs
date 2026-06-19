// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
use std::iter;
use std::sync::Arc;
use std::sync::LazyLock;

use bytes::Bytes;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::pin_mut;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::PType::I32;
use vortex_array::dtype::StructFields;
use vortex_array::expr::and;
use vortex_array::expr::cast;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::extension::datetime::TimestampOptions;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSession;
use vortex_layout::Layout;
use vortex_layout::layouts::zoned::LegacyStats;
use vortex_layout::layouts::zoned::Zoned;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::V1_FOOTER_FBS_SIZE;
use crate::VERSION;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
use crate::footer::SegmentSpec;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    crate::register_default_encodings(&session);

    session
});

#[tokio::test]
async fn test_eof_values() {
    // this test exists as a reminder to think about whether we should increment the version
    // when we change the footer
    assert_eq!(VERSION, 1);
    assert_eq!(V1_FOOTER_FBS_SIZE, 32);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_simple() {
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let stream = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap();
    pin_mut!(stream);

    let mut row_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        row_count += array.len();
    }

    assert_eq!(row_count, 8);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_round_trip_many_types() {
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
    let dtype = st.dtype().clone();
    let mut buf = ByteBufferMut::empty();

    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let chunks: Vec<_> = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    let read = ChunkedArray::try_new(chunks, dtype).unwrap();

    assert_eq!(read.len(), 3);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_simple_with_spawn() {
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    assert!(!buf.is_empty());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_projection() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();
    let array = file
        .scan()
        .unwrap()
        .with_projection(select(["strings"], root()))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructFields::new(["strings"].into(), vec![strings_dtype]),
            Nullability::NonNullable,
        )
    );

    let actual = array
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(0)
        .clone();
    let expected = VarBinArray::from(strings_expected.to_vec()).into_array();
    assert_arrays_eq!(actual, expected);

    let array = file
        .scan()
        .unwrap()
        .with_projection(select(["numbers"], root()))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructFields::new(["numbers"].into(), vec![numbers_dtype]),
            Nullability::NonNullable,
        )
    );

    let actual = array
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(0)
        .clone();
    let expected = Buffer::copy_from(numbers_expected).into_array();
    assert_arrays_eq!(actual, expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn unequal_batches() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let stream = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap();
    pin_mut!(stream);

    let mut item_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        item_count += array.len();

        let numbers = array
            .execute::<StructArray>(&mut ctx)
            .unwrap()
            .unmasked_field_by_name("numbers")
            .unwrap()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(numbers.ptype(), PType::U32);
    }
    assert_eq!(item_count, 10);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn write_chunked() {
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
    SESSION
        .write_options()
        .write(&mut buf, chunked_st.into_array().to_array_stream())
        .await
        .unwrap();

    let stream = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap();
    pin_mut!(stream);

    let mut array_len: usize = 0;
    while let Some(array) = stream.next().await {
        array_len += array.unwrap().len();
    }
    assert_eq!(array_len, 48);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_empty_varbin_array_roundtrip() {
    let empty = VarBinArray::from(Vec::<&str>::new()).into_array();

    let st = StructArray::from_fields(&[("a", empty)]).unwrap();
    let dtype = st.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let result = file
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), &dtype);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn issue_5385_filter_casted_column() {
    let array = StructArray::try_from_iter([("x", buffer![1u8, 2, 3, 4, 5])])
        .unwrap()
        .into_array();

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await
        .unwrap();

    let result = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(eq(
            cast(
                get_item("x", root()),
                DType::Primitive(PType::U16, Nullability::NonNullable),
            ),
            lit(1u16),
        ))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_arrays_eq!(
        result,
        StructArray::try_from_iter([("x", buffer![1u8])]).unwrap()
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let result: Vec<_> = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(eq(get_item("name", root()), lit("Joseph")))
        .into_array_stream()
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(0)
        .clone();
    let names_expected =
        VarBinArray::from_iter(vec![Some("Joseph")], DType::Utf8(Nullability::Nullable))
            .into_array();
    assert_arrays_eq!(names_actual, names_expected);

    let ages_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(1)
        .clone();
    let ages_expected = PrimitiveArray::from_option_iter([Some(25i32)]).into_array();
    assert_arrays_eq!(ages_actual, ages_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_or() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let result: Vec<_> = SESSION
        .open_options()
        .open_buffer(buf)
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
        .into_array_stream()
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(0)
        .clone();
    let names_expected = VarBinArray::from_iter(
        vec![Some("Joseph"), Some("Angela")],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    assert_arrays_eq!(names_actual, names_expected);

    let ages_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(1)
        .clone();
    let ages_expected = PrimitiveArray::from_option_iter([Some(25i32), None]).into_array();
    assert_arrays_eq!(ages_actual, ages_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_and() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let result: Vec<_> = SESSION
        .open_options()
        .open_buffer(buf)
        .unwrap()
        .scan()
        .unwrap()
        .with_filter(and(
            gt(get_item("age", root()), lit(21)),
            lt_eq(get_item("age", root()), lit(33)),
        ))
        .into_array_stream()
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(0)
        .clone();
    let names_expected = VarBinArray::from_iter(
        vec![Some("Joseph"), None],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    assert_arrays_eq!(names_actual, names_expected);

    let ages_actual = result[0]
        .clone()
        .execute::<StructArray>(&mut ctx)
        .unwrap()
        .unmasked_field(1)
        .clone();
    let ages_expected = PrimitiveArray::from_option_iter([Some(25i32), Some(31i32)]).into_array();
    assert_arrays_eq!(ages_actual, ages_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_simple() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, expected_array.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    // test no indices
    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::<u64>::empty())
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0_u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();
    let actual_kept_numbers_array = actual_kept_array
        .unmasked_field(0)
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();

    let expected_kept_numbers: Vec<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .collect();
    let expected_array = Buffer::copy_from(&expected_kept_numbers).into_array();
    assert_arrays_eq!(actual_kept_numbers_array, expected_array);

    // test all indices
    let actual_array = file
        .scan()
        .unwrap()
        .with_row_indices((0u64..500).collect::<Buffer<_>>())
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();
    let actual_numbers_array = actual_array.unmasked_field(0).clone();
    let expected_array = Buffer::copy_from(&expected_numbers).into_array();
    assert_arrays_eq!(actual_numbers_array, expected_array);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_on_two_columns() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let kept_indices = [0_u64, 3, 7];
    let array = file
        .scan()
        .unwrap()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    let strings_actual = array.unmasked_field(0).clone();
    let strings_expected_vec: Vec<&str> = kept_indices
        .iter()
        .map(|&x| strings_expected[x as usize])
        .collect();
    let strings_expected_array = VarBinArray::from(strings_expected_vec).into_array();
    assert_arrays_eq!(strings_actual, strings_expected_array);

    let numbers_actual = array.unmasked_field(1).clone();
    let numbers_expected_vec: Vec<u32> = kept_indices
        .iter()
        .map(|&x| numbers_expected[x as usize])
        .collect();
    let numbers_expected_array = Buffer::copy_from(&numbers_expected_vec).into_array();
    assert_arrays_eq!(numbers_actual, numbers_expected_array);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_and_with_row_filter_simple() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, expected_array.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices(Buffer::empty())
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    let actual_kept_numbers_array = actual_kept_array
        .unmasked_field(0)
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();

    let expected_kept_numbers: Buffer<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .filter(|&x| x > 50)
        .collect();
    let expected_array = expected_kept_numbers.into_array();
    assert_arrays_eq!(actual_kept_numbers_array, expected_array);

    // test all indices
    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(gt(get_item("numbers", root()), lit(50_i16)))
        .with_row_indices((0..500).collect::<Buffer<_>>())
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    let actual_numbers_array = actual_array.unmasked_field(0).clone();
    let expected_filtered: Buffer<i16> = expected_numbers
        .iter()
        .filter(|&&x| x > 50)
        .cloned()
        .collect();
    let expected_numbers_array = expected_filtered.into_array();
    assert_arrays_eq!(actual_numbers_array, expected_numbers_array);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string_chunked() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(eq(get_item("name", root()), lit("Joseph")))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_array.len(), 1);
    let names_actual = actual_array.unmasked_field(0).clone();
    let names_expected =
        VarBinArray::from_iter(vec![Some("Joseph")], DType::Utf8(Nullability::Nullable))
            .into_array();
    assert_arrays_eq!(names_actual, names_expected);

    let ages_actual = actual_array.unmasked_field(1).clone();
    let ages_expected = PrimitiveArray::from_option_iter([Some(25i32)]).into_array();
    assert_arrays_eq!(ages_actual, ages_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_pruning_with_or() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let actual_array = file
        .scan()
        .unwrap()
        .with_filter(or(
            lt_eq(get_item("letter", root()), lit("J")),
            lt(get_item("number", root()), lit(25)),
        ))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_eq!(actual_array.len(), 10);
    let letters_actual = actual_array.unmasked_field(0).clone();
    let letters_expected = VarBinViewArray::from_iter_nullable_str([
        Some("A".to_owned()),
        Some("B".to_owned()),
        Some("D".to_owned()),
        Some("G".to_owned()),
        Some("I".to_owned()),
        Some("J".to_owned()),
        None,
        Some("L".to_owned()),
        None,
        Some("P".to_owned()),
    ])
    .into_array();
    assert_arrays_eq!(letters_actual, letters_expected);

    let numbers_actual = actual_array.unmasked_field(1).clone();
    let numbers_expected = PrimitiveArray::from_option_iter([
        Some(25_i32),
        Some(31),
        None,
        Some(4),
        Some(18),
        None,
        Some(21),
        Some(10),
        Some(15),
        Some(22),
    ])
    .into_array();
    assert_arrays_eq!(numbers_actual, numbers_expected);
}

#[tokio::test]
async fn test_repeated_projection() {
    let mut ctx = SESSION.create_execution_ctx();
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
    SESSION
        .write_options()
        .write(&mut buf, single_column_array.into_array().to_array_stream())
        .await
        .unwrap();

    let file = SESSION.open_options().open_buffer(buf).unwrap();

    let actual = file
        .scan()
        .unwrap()
        .with_projection(select(["strings", "strings"], root()))
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .execute::<StructArray>(&mut ctx)
        .unwrap();

    assert_arrays_eq!(actual, expected);
}

async fn chunked_file() -> VortexResult<VortexFile> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();

    let mut writer = vec![];
    SESSION
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    let buffer: Bytes = writer.into();
    SESSION.open_options().open_buffer(buffer)
}

#[tokio::test]
async fn basic_file_roundtrip() -> VortexResult<()> {
    let vxf = chunked_file().await?;
    let result = vxf.scan()?.into_array_stream()?.read_all().await?;

    let expected = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8].into_array();
    assert_arrays_eq!(result, expected);

    Ok(())
}

#[tokio::test]
async fn file_excluding_dtype() -> VortexResult<()> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();
    let dtype = array.dtype().clone();

    let mut writer = vec![];
    SESSION
        .write_options()
        .exclude_dtype()
        .write(&mut writer, array.to_array_stream())
        .await?;
    let buffer: Bytes = writer.into();

    // Fail to open without DType.
    let vxf = SESSION.open_options().open_buffer(buffer.clone());
    assert!(vxf.is_err(), "Opening without DType should fail");

    let vxf = SESSION
        .open_options()
        .with_dtype(dtype.clone())
        .open_buffer(buffer)?;
    assert_eq!(vxf.dtype(), &dtype);
    assert_eq!(vxf.row_count(), 9);

    Ok(())
}

#[tokio::test]
async fn file_take() -> VortexResult<()> {
    let vxf = chunked_file().await?;
    let result = vxf
        .scan()?
        .with_row_indices(buffer![0, 1, 8])
        .into_array_stream()?
        .read_all()
        .await?;

    let expected = buffer![0i32, 1, 8].into_array();
    assert_arrays_eq!(result, expected);

    Ok(())
}

#[tokio::test]
#[should_panic(
    expected = "FileStatsAccumulator temporarily does not support nullable top-level structs"
)]
async fn write_nullable_top_level_struct() {
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
    SESSION
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await
        .unwrap();
}

async fn round_trip(
    array: &ArrayRef,
    f: impl Fn(ScanBuilder<ArrayRef>) -> VortexResult<ScanBuilder<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let mut writer = vec![];
    SESSION
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    let buffer: Bytes = writer.into();

    let vxf = SESSION
        .open_options()
        .with_dtype(array.dtype().clone())
        .open_buffer(buffer)?;

    assert_eq!(vxf.dtype(), array.dtype());
    assert_eq!(vxf.row_count(), array.len() as u64);

    f(vxf.scan()?)?.into_array_stream()?.read_all().await
}

#[tokio::test]
async fn write_nullable_nested_struct() -> VortexResult<()> {
    let nested_dtype = DType::struct_(
        [(
            "nested_field",
            DType::Primitive(PType::F16, Nullability::Nullable),
        )],
        Nullability::Nullable,
    );

    let struct_ = ConstantArray::new(Scalar::null(nested_dtype.clone()), 3).into_array();

    let array = StructArray::try_new(
        ["struct"].into(),
        vec![struct_.into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let result = round_trip(&array, Ok)
        .await?
        .execute::<StructArray>(&mut ctx)?;

    assert_eq!(result.len(), 3);
    assert_eq!(result.struct_fields().nfields(), 1);
    assert!(result.all_valid(&mut ctx)?);

    let nested_struct = result
        .unmasked_field_by_name("struct")?
        .clone()
        .execute::<StructArray>(&mut ctx)?;
    assert_eq!(nested_struct.dtype(), &nested_dtype);
    assert_eq!(nested_struct.len(), 3);
    assert!(nested_struct.all_invalid(&mut ctx)?);

    Ok(())
}

#[tokio::test]
async fn scan_empty_fields() -> VortexResult<()> {
    let array = (0..10000).collect::<PrimitiveArray>();

    let result = round_trip(&array.clone().into_array(), |scan| {
        Ok(scan.with_projection(Pack.new_expr(
            PackOptions {
                names: Default::default(),
                nullability: Nullability::Nullable,
            },
            [],
        )))
    })
    .await?;

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
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await?;

    let file = SESSION.open_options().open_buffer(buf)?;
    let stream = file.scan().unwrap().into_array_stream()?;
    let array = stream.read_all().await?;

    assert_eq!(array.len(), 8);

    Ok(())
}

#[tokio::test]
async fn test_array_stream_no_double_dict_encode() -> VortexResult<()> {
    let num_vals = 2048;
    let mut values = Vec::<i64>::with_capacity(num_vals);
    values.extend(iter::repeat_n(0, num_vals / 2));
    values.extend(iter::repeat_n(1, num_vals / 2));

    let array = PrimitiveArray::from_iter(values).into_array();
    let mut buf = Vec::new();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;
    let file = SESSION.open_options().open_buffer(buf)?;
    let read_array = file.scan()?.into_array_stream()?.read_all().await?;

    let dict = read_array
        .as_opt::<Dict>()
        .expect("expected root to be dictionary");
    assert!(
        !dict.codes().is::<Dict>(),
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
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push(st.clone()).await?;
    let summary = writer.finish().await?;

    assert_eq!(summary.row_count(), 4);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 4);
    assert_eq!(result.dtype(), &dtype);

    Ok(())
}

#[tokio::test]
async fn test_writer_multiple_pushes() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let chunk1 =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3].into_array())])?.into_array();
    let chunk2 =
        StructArray::from_fields(&[("numbers", buffer![4u32, 5, 6].into_array())])?.into_array();
    let chunk3 =
        StructArray::from_fields(&[("numbers", buffer![7u32, 8, 9].into_array())])?.into_array();

    let dtype = chunk1.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push(chunk1).await?;
    writer.push(chunk2).await?;
    writer.push(chunk3).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 9);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 9);
    let numbers = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("numbers")?
        .clone();
    let expected = buffer![1u32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    assert_arrays_eq!(numbers, expected);

    Ok(())
}

#[tokio::test]
async fn test_writer_push_stream() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let chunk1 =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3].into_array())])?.into_array();
    let chunk2 =
        StructArray::from_fields(&[("numbers", buffer![4u32, 5, 6].into_array())])?.into_array();

    let dtype = chunk1.dtype().clone();

    let stream = futures::stream::iter(vec![Ok(chunk1), Ok(chunk2)]);
    let sendable_stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype.clone(), stream));

    let mut buf = ByteBufferMut::empty();
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push_stream(sendable_stream).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 6);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 6);
    let numbers = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("numbers")?
        .clone();
    let expected = buffer![1u32, 2, 3, 4, 5, 6].into_array();
    assert_arrays_eq!(numbers, expected);

    Ok(())
}

#[tokio::test]
async fn test_writer_bytes_written() -> VortexResult<()> {
    let array = StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3, 4, 5].into_array())])?
        .into_array();
    let dtype = array.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = SESSION.write_options().writer(&mut buf, dtype);

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
    let mut ctx = SESSION.create_execution_ctx();
    let empty = StructArray::from_fields(&[(
        "numbers",
        PrimitiveArray::new::<u32>(buffer![], Validity::NonNullable).into_array(),
    )])?
    .into_array();
    let non_empty =
        StructArray::from_fields(&[("numbers", buffer![1u32, 2].into_array())])?.into_array();

    let dtype = empty.dtype().clone();

    let mut buf = ByteBufferMut::empty();
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push(empty.clone()).await?;
    writer.push(non_empty).await?;
    writer.push(empty).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 2);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 2);
    let numbers = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("numbers")?
        .clone();
    let expected = buffer![1u32, 2].into_array();
    assert_arrays_eq!(numbers, expected);

    Ok(())
}

#[tokio::test]
async fn test_writer_mixed_push_and_stream() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push(chunk1).await?;
    writer.push_stream(sendable_stream).await?;
    writer.push(chunk3).await?;

    let summary = writer.finish().await?;
    assert_eq!(summary.row_count(), 6);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 6);
    let numbers = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("numbers")?
        .clone();
    let expected = buffer![1u32, 2, 3, 4, 5, 6].into_array();
    assert_arrays_eq!(numbers, expected);

    Ok(())
}

#[tokio::test]
async fn test_writer_with_complex_types() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut writer = SESSION.write_options().writer(&mut buf, dtype.clone());

    writer.push(chunk).await?;
    let footer = writer.finish().await?;

    assert_eq!(footer.row_count(), 3);

    let file = SESSION.open_options().open_buffer(buf)?;
    let result = file.scan()?.into_array_stream()?.read_all().await?;

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype(), &dtype);

    let strings_field = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("strings")
        .cloned()?;
    let strings = strings_field
        .execute::<VarBinViewArray>(&mut ctx)?
        .with_iterator(|iter| {
            iter.map(|s| s.map(|st| unsafe { String::from_utf8_unchecked(st.to_vec()) }))
                .collect::<Vec<_>>()
        });
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
    let mut writer = SESSION
        .write_options()
        .with_file_statistics(PRUNING_STATS.to_vec())
        .writer(&mut buf, array.dtype().clone());

    writer.push(array).await?;
    let summary = writer.finish().await?;

    assert!(summary.footer().statistics().is_some());
    assert_eq!(summary.row_count(), 5);

    Ok(())
}

#[tokio::test]
async fn timestamp_unit_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    // Write file with MILLISECONDS timestamps
    let ts_array = PrimitiveArray::from_iter(vec![1704067200000i64, 1704153600000, 1704240000000])
        .into_array();
    let temporal = TemporalArray::new_timestamp(ts_array, TimeUnit::Milliseconds, None);

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, temporal.into_array().to_array_stream())
        .await?;

    // Read with SECONDS filter scalar
    let filter_expr = gt(
        root(),
        lit(Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            Scalar::from(1704153600i64),
        )),
    );

    let mut stream = SESSION
        .open_options()
        .open_buffer(buf)?
        .scan()?
        .with_filter(filter_expr)
        .into_array_stream()?;

    let result = stream.try_next().await;

    assert!(result.is_err());

    Ok(())
}

/// Regression test: filtering a milliseconds timestamp column with a seconds scalar should
/// always error, regardless of how the internal children of `DateTimePartsArray` are encoded.
///
/// This test forces `ConstantArray` encoding for the seconds/subseconds children by using a
/// compressor with Dict excluded (which triggers distinct-value computation, letting
/// `ConstantScheme` win for `[0, 0, 0]`). The scanner should still detect the time unit
/// mismatch and error, not silently return wrong results.
#[tokio::test]
async fn timestamp_unit_mismatch_errors_with_constant_children()
-> Result<(), Box<dyn std::error::Error>> {
    // Build a compressor where ConstantScheme wins for [0, 0, 0] by including Dict
    // (which enables distinct-value computation).
    let compressor = vortex_btrblocks::BtrBlocksCompressor::default();

    // Write file with MILLISECONDS timestamps using this compressor.
    let ts_array = PrimitiveArray::from_iter(vec![1704067200000i64, 1704153600000, 1704240000000])
        .into_array();
    let temporal = TemporalArray::new_timestamp(ts_array, TimeUnit::Milliseconds, None);

    let strategy = crate::strategy::WriteStrategyBuilder::default()
        .with_compressor(compressor)
        .build();

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut buf, temporal.into_array().to_array_stream())
        .await?;

    // Read with SECONDS filter scalar — should error due to time unit mismatch.
    let filter_expr = gt(
        root(),
        lit(Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            Scalar::from(1704153600i64),
        )),
    );

    let stream = SESSION
        .open_options()
        .open_buffer(buf)?
        .scan()?
        .with_filter(filter_expr)
        .into_array_stream()?;

    let results = stream.try_collect::<Vec<_>>().await;

    assert!(
        results.is_err(),
        "Expected error from timestamp unit mismatch (ms vs s), but got {} results. \
         This indicates the scanner silently applied the filter incorrectly when \
         DateTimePartsArray children use ConstantArray encoding.",
        results.unwrap().len()
    );

    Ok(())
}

/// Collect all segment byte offsets reachable from a layout node.
fn collect_segment_offsets(layout: &dyn Layout, segment_specs: &[SegmentSpec]) -> Vec<u64> {
    let mut result = Vec::new();
    collect_segment_offsets_inner(layout, segment_specs, &mut result);
    result
}

fn collect_segment_offsets_inner(
    layout: &dyn Layout,
    segment_specs: &[SegmentSpec],
    result: &mut Vec<u64>,
) {
    for seg_id in layout.segment_ids() {
        result.push(segment_specs[*seg_id as usize].offset);
    }
    for child in layout.children().unwrap() {
        collect_segment_offsets_inner(child.as_ref(), segment_specs, result);
    }
}

/// Assert that all offsets in `before` are less than all offsets in `after`.
fn assert_offsets_ordered(before: &[u64], after: &[u64], context: &str) {
    if let (Some(&max_before), Some(&min_after)) = (before.iter().max(), after.iter().min()) {
        assert!(
            max_before < min_after,
            "{context}: expected all 'before' offsets < all 'after' offsets, \
             but max before = {max_before} >= min after = {min_after}"
        );
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_segment_ordering_dict_codes_before_values() -> VortexResult<()> {
    // Create low-cardinality strings to trigger dict encoding, plus an integer column.
    let n = 100_000;
    let values: Vec<&str> = (0..n).map(|i| ["alpha", "beta", "gamma"][i % 3]).collect();
    let strings = VarBinArray::from(values).into_array();
    let numbers = PrimitiveArray::from_iter(0..n as i32).into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();

    let mut buf = ByteBufferMut::empty();
    let summary = SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await?;

    let footer = summary.footer();
    let segment_specs = footer.segment_map();
    let root = footer.layout();

    // Walk the layout tree and find all dict layouts.
    // Verify codes segments come before values segments in byte order within each run.
    fn check_dict_ordering(layout: &dyn Layout, segment_specs: &[SegmentSpec]) {
        if layout.encoding_id().as_ref() == "vortex.dict" {
            // child 0 = values, child 1 = codes
            let values_offsets =
                collect_segment_offsets(layout.child(0).unwrap().as_ref(), segment_specs);
            let codes_offsets =
                collect_segment_offsets(layout.child(1).unwrap().as_ref(), segment_specs);

            assert_offsets_ordered(
                &codes_offsets,
                &values_offsets,
                "dict: codes should come before values",
            );
        }

        for child in layout.children().unwrap() {
            check_dict_ordering(child.as_ref(), segment_specs);
        }
    }

    check_dict_ordering(root.as_ref(), segment_specs);

    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_segment_ordering_zonemaps_after_data() -> VortexResult<()> {
    // Create a multi-column struct with enough rows to produce zone maps.
    let n = 100_000;
    let values: Vec<&str> = (0..n).map(|i| ["alpha", "beta", "gamma"][i % 3]).collect();
    let strings = VarBinArray::from(values).into_array();
    let numbers = PrimitiveArray::from_iter(0..n as i32).into_array();
    let floats = PrimitiveArray::from_iter((0..n).map(|i| i as f64 * 0.1)).into_array();

    let st = StructArray::from_fields(&[
        ("strings", strings),
        ("numbers", numbers),
        ("floats", floats),
    ])
    .unwrap();

    let mut buf = ByteBufferMut::empty();
    let summary = SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await?;

    let footer = summary.footer();
    let segment_specs = footer.segment_map();
    let root = footer.layout();

    // Find all zoned layouts and verify data segments come before zone map segments.
    fn check_zoned_ordering(layout: &dyn Layout, segment_specs: &[SegmentSpec]) {
        if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
            // child 0 = data, child 1 = zones
            let data_offsets =
                collect_segment_offsets(layout.child(0).unwrap().as_ref(), segment_specs);
            let zones_offsets =
                collect_segment_offsets(layout.child(1).unwrap().as_ref(), segment_specs);

            assert_offsets_ordered(
                &data_offsets,
                &zones_offsets,
                "zoned: data should come before zones",
            );
        }

        for child in layout.children().unwrap() {
            check_zoned_ordering(child.as_ref(), segment_specs);
        }
    }

    check_zoned_ordering(root.as_ref(), segment_specs);

    // Additionally: all zone map segments across all columns should appear after
    // all data segments across all columns.
    let mut all_data_offsets = Vec::new();
    let mut all_zones_offsets = Vec::new();

    fn collect_all_zoned(
        layout: &dyn Layout,
        segment_specs: &[SegmentSpec],
        all_data: &mut Vec<u64>,
        all_zones: &mut Vec<u64>,
    ) {
        if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
            // child 0 = data, child 1 = zones
            all_data.extend(collect_segment_offsets(
                layout.child(0).unwrap().as_ref(),
                segment_specs,
            ));
            all_zones.extend(collect_segment_offsets(
                layout.child(1).unwrap().as_ref(),
                segment_specs,
            ));
            return;
        }
        for child in layout.children().unwrap() {
            collect_all_zoned(child.as_ref(), segment_specs, all_data, all_zones);
        }
    }

    collect_all_zoned(
        root.as_ref(),
        segment_specs,
        &mut all_data_offsets,
        &mut all_zones_offsets,
    );

    assert_offsets_ordered(
        &all_data_offsets,
        &all_zones_offsets,
        "global: all data segments should come before all zone map segments",
    );

    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_can_prune_composite_predicates() -> VortexResult<()> {
    // Regression test for `can_prune` after `ScalarFnConstantRule` was removed
    // (#7575): composite falsification trees no longer constant-fold during
    // execution, so `can_prune` must read the one-row evaluated result instead
    // of requiring a `Columnar::Constant`. `Eq` is affected too: its
    // falsification is internally `or(min > lit, lit > max)`.
    let st = StructArray::from_fields(&[
        ("age", buffer![15i32, 18, 22, 25].into_array()),
        ("price", buffer![120i32, 130, 140, 150].into_array()),
    ])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, st.into_array().to_array_stream())
        .await?;
    let file = SESSION.open_options().open_buffer(buf)?;

    // Bare comparisons: falsified directly by min/max stats.
    assert!(file.can_prune(&gt(col("age"), lit(30)))?);
    assert!(file.can_prune(&lt(col("price"), lit(100)))?);

    // Composite predicates whose falsifications are boolean trees.
    assert!(file.can_prune(&and(gt(col("age"), lit(30)), lt(col("price"), lit(100))))?);
    assert!(file.can_prune(&or(gt(col("age"), lit(30)), lt(col("age"), lit(10))))?);
    assert!(file.can_prune(&eq(col("age"), lit(5)))?);

    // Non-falsifiable controls: rows may match, so pruning must refuse.
    assert!(!file.can_prune(&gt(col("age"), lit(20)))?);
    assert!(!file.can_prune(&eq(col("age"), lit(18)))?);
    assert!(!file.can_prune(&and(gt(col("age"), lit(20)), gt(col("price"), lit(100))))?);

    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn repro_8166_binary_gt_all_ff_max() -> VortexResult<()> {
    use vortex_buffer::ByteBuffer;

    let mut ctx = SESSION.create_execution_ctx();

    let empty: Vec<u8> = vec![];
    let chunk0: Vec<Vec<u8>> = vec![
        vec![0x1d, 0x00],
        empty.clone(),
        vec![0x1d, 0x10, 0x9d, 0x08],
        empty.clone(),
        empty.clone(),
        empty.clone(),
        empty.clone(),
        empty.clone(),
        empty.clone(),
    ];
    let chunk1: Vec<Vec<u8>> = vec![
        empty.clone(),
        empty.clone(),
        vec![0x40],
        empty.clone(),
        empty.clone(),
        empty.clone(),
        empty.clone(),
        empty.clone(),
        vec![0x24],
        vec![0x43, 0xff],
    ];
    let mut big = vec![0xffu8; 112];
    big[89] = 0x03;
    let mut chunk2: Vec<Vec<u8>> = vec![empty.clone(); 10];
    chunk2[8] = big;

    let bin = DType::Binary(Nullability::NonNullable);
    let mk_struct = |vals: Vec<Vec<u8>>| -> VortexResult<ArrayRef> {
        let yyw = VarBinArray::from_vec(vals, bin.clone()).into_array();
        Ok(StructArray::from_fields(&[("yyw", yyw)])?.into_array())
    };
    let array =
        ChunkedArray::from_iter([mk_struct(chunk0)?, mk_struct(chunk1)?, mk_struct(chunk2)?])
            .into_array();

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;

    let mut literal = vec![0x6fu8; 5];
    literal.extend(iter::repeat_n(0xffu8, 57));
    literal.push(0x98);
    assert_eq!(literal.len(), 63);

    let filter = gt(
        get_item("yyw", root()),
        lit(Scalar::binary(
            ByteBuffer::from(literal),
            Nullability::NonNullable,
        )),
    );

    let result = SESSION
        .open_options()
        .open_buffer(buf)?
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .read_all()
        .await?
        .execute::<StructArray>(&mut ctx)?;

    assert_eq!(result.len(), 1);
    Ok(())
}
