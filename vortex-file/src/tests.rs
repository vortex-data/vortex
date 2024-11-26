use std::collections::BTreeSet;
use std::iter;
use std::sync::{Arc, RwLock};

use futures::StreamExt;
use itertools::Itertools;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::compute::unary::scalar_at;
use vortex_array::validity::Validity;
use vortex_array::variants::{PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_buffer::Buffer;
use vortex_dtype::field::Field;
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::vortex_panic;
use vortex_expr::{BinaryExpr, Column, Literal, Operator};

use crate::builder::initial_read::{read_initial_bytes, read_layout_from_initial};
use crate::write::VortexFileWriter;
use crate::{
    LayoutDeserializer, LayoutMessageCache, Projection, RelativeLayoutCache, RowFilter, Scan,
    VortexReadBuilder, V1_FOOTER_FBS_SIZE, VERSION,
};

#[test]
fn test_eof_values() {
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
        PrimitiveArray::from(vec![1u32, 2, 3, 4]).into_array(),
        PrimitiveArray::from(vec![5u32, 6, 7, 8]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    let mut stream = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .build()
        .await
        .unwrap();
    let mut batch_count = 0;
    let mut row_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        batch_count += 1;
        row_count += array.len();
    }

    assert_eq!(batch_count, 2);
    assert_eq!(row_count, 8);
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
        PrimitiveArray::from(vec![1u32, 2, 3, 4]).into_array(),
        PrimitiveArray::from(vec![5u32, 6, 7, 8]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = Vec::new();

    let written = tokio::spawn(async move {
        let mut writer = VortexFileWriter::new(buf);
        writer = writer.write_array_columns(st.into_array()).await.unwrap();
        Buffer::from(writer.finalize().await.unwrap())
    })
    .await
    .unwrap();

    assert!(!written.is_empty());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_splits() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        PrimitiveArray::from(vec![1u32, 2, 3]).into_array(),
        PrimitiveArray::from(vec![4u32, 5, 6]).into_array(),
        PrimitiveArray::from(vec![7u32, 8]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let len = st.len();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    let initial_read = read_initial_bytes(&written, written.len() as u64)
        .await
        .unwrap();
    let layout_serde = LayoutDeserializer::default();

    let dtype = Arc::new(initial_read.lazy_dtype().unwrap());
    let cache = Arc::new(RwLock::new(LayoutMessageCache::new()));

    let layout_reader = read_layout_from_initial(
        &initial_read,
        &layout_serde,
        Scan::new(None),
        RelativeLayoutCache::new(cache, dtype),
    )
    .unwrap();

    let mut splits = BTreeSet::new();
    layout_reader.add_splits(0, &mut splits).unwrap();
    splits.insert(len);
    assert_eq!(splits, BTreeSet::from([0, 3, 5, 6, 8]));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_projection() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();
    let strings_dtype = strings.dtype().clone();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        PrimitiveArray::from(numbers_expected[..4].to_vec()).into_array(),
        PrimitiveArray::from(numbers_expected[4..].to_vec()).into_array(),
    ])
    .into_array();
    let numbers_dtype = numbers.dtype().clone();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    let array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::new([0]))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructDType::new(vec!["strings".into()].into(), vec![strings_dtype.clone()]),
            Nullability::NonNullable,
        )
    );

    let actual = array
        .into_struct()
        .unwrap()
        .field(0)
        .unwrap()
        .into_varbinview()
        .unwrap()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(actual, strings_expected);

    let array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::Flat(vec![Field::Name("strings".to_string())]))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructDType::new(vec!["strings".into()].into(), vec![strings_dtype.clone()]),
            Nullability::NonNullable,
        )
    );

    let actual = array
        .into_struct()
        .unwrap()
        .field(0)
        .unwrap()
        .into_varbinview()
        .unwrap()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(actual, strings_expected);

    let array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::new([1]))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructDType::new(vec!["numbers".into()].into(), vec![numbers_dtype.clone()]),
            Nullability::NonNullable,
        )
    );

    let primitive_array = array
        .into_struct()
        .unwrap()
        .field(0)
        .unwrap()
        .into_primitive()
        .unwrap();
    let actual = primitive_array.maybe_null_slice::<u32>();
    assert_eq!(actual, numbers_expected);

    let array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::Flat(vec![Field::Name("numbers".to_string())]))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            StructDType::new(vec!["numbers".into()].into(), vec![numbers_dtype.clone()]),
            Nullability::NonNullable,
        )
    );

    let primitive_array = array
        .into_struct()
        .unwrap()
        .field(0)
        .unwrap()
        .into_primitive()
        .unwrap();
    let actual = primitive_array.maybe_null_slice::<u32>();
    assert_eq!(actual, numbers_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn unequal_batches() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "bob"]).into_array(),
        VarBinArray::from(vec!["baz", "ab", "foo", "bar", "baz", "alice"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        PrimitiveArray::from(vec![1u32, 2, 3, 4, 5]).into_array(),
        PrimitiveArray::from(vec![6u32, 7, 8, 9, 10]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    let mut stream = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .build()
        .await
        .unwrap();
    let mut batch_count = 0;
    let mut item_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        item_count += array.len();
        batch_count += 1;

        let numbers = array.with_dyn(|a| a.as_struct_array_unchecked().field_by_name("numbers"));

        if let Some(numbers) = numbers {
            let numbers = numbers.into_primitive().unwrap();
            assert_eq!(numbers.ptype(), PType::U32);
        } else {
            vortex_panic!("Expected column doesn't exist")
        }
    }
    assert_eq!(item_count, 10);
    assert_eq!(batch_count, 3);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn write_chunked() {
    let strings = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array();
    let string_dtype = strings.dtype().clone();
    let strings_chunked =
        ChunkedArray::try_new(iter::repeat(strings).take(4).collect(), string_dtype)
            .unwrap()
            .into_array();
    let numbers = PrimitiveArray::from(vec![1u32, 2, 3, 4]).into_array();
    let numbers_dtype = numbers.dtype().clone();
    let numbers_chunked =
        ChunkedArray::try_new(iter::repeat(numbers).take(4).collect(), numbers_dtype)
            .unwrap()
            .into_array();
    let st = StructArray::try_new(
        ["strings".into(), "numbers".into()].into(),
        vec![strings_chunked, numbers_chunked],
        16,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let st_dtype = st.dtype().clone();

    let chunked_st = ChunkedArray::try_new(iter::repeat(st).take(3).collect(), st_dtype)
        .unwrap()
        .into_array();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(chunked_st).await.unwrap();

    let written = Buffer::from(writer.finalize().await.unwrap());
    let mut reader = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .build()
        .await
        .unwrap();
    let mut array_len: usize = 0;
    while let Some(array) = reader.next().await {
        array_len += array.unwrap().len();
    }
    assert_eq!(array_len, 48);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string() {
    let names_orig = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let ages_orig =
        PrimitiveArray::from_nullable_vec(vec![Some(25), Some(31), None, Some(57), None])
            .into_array();
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names_orig, ages_orig],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut writer = VortexFileWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();

    let written = Buffer::from(writer.finalize().await.unwrap());
    let mut reader = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            Column::new_expr(Field::from("name")),
            Operator::Eq,
            Literal::new_expr("Joseph".into()),
        )))
        .build()
        .await
        .unwrap();

    let mut result = Vec::new();
    while let Some(array) = reader.next().await {
        result.push(array.unwrap());
    }
    assert_eq!(result.len(), 1);
    let names = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(0))
        .unwrap();
    assert_eq!(
        names
            .into_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(1))
        .unwrap();
    assert_eq!(
        ages.into_primitive().unwrap().maybe_null_slice::<i32>(),
        vec![25]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_or() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_nullable_vec(vec![Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names.to_array(), ages.to_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut writer = VortexFileWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());
    let mut reader = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            BinaryExpr::new_expr(
                Column::new_expr(Field::from("name")),
                Operator::Eq,
                Literal::new_expr("Angela".into()),
            ),
            Operator::Or,
            BinaryExpr::new_expr(
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("age")),
                    Operator::Gte,
                    Literal::new_expr(20.into()),
                ),
                Operator::And,
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("age")),
                    Operator::Lte,
                    Literal::new_expr(30.into()),
                ),
            ),
        )))
        .build()
        .await
        .unwrap();

    let mut result = Vec::new();
    while let Some(array) = reader.next().await {
        result.push(array.unwrap());
    }
    assert_eq!(result.len(), 1);
    let names = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(0))
        .unwrap();
    assert_eq!(
        names
            .into_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string(), "Angela".to_string()]
    );
    let ages = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(1))
        .unwrap();
    assert_eq!(
        ages.into_primitive()
            .unwrap()
            .with_iterator(|iter| iter.map(|x| x.cloned()).collect::<Vec<_>>())
            .unwrap(),
        vec![Some(25), None]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_and() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_nullable_vec(vec![Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names.to_array(), ages.to_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut writer = VortexFileWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());
    let mut reader = VortexReadBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            BinaryExpr::new_expr(
                Column::new_expr(Field::from("age")),
                Operator::Gt,
                Literal::new_expr(21.into()),
            ),
            Operator::And,
            BinaryExpr::new_expr(
                Column::new_expr(Field::from("age")),
                Operator::Lte,
                Literal::new_expr(33.into()),
            ),
        )))
        .build()
        .await
        .unwrap();

    let mut result = Vec::new();
    while let Some(array) = reader.next().await {
        result.push(array.unwrap());
    }
    assert_eq!(result.len(), 1);
    let names = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(0))
        .unwrap();
    assert_eq!(
        names
            .into_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .map(|s| s.map(|st| unsafe { String::from_utf8_unchecked(st.to_vec()) }))
                .collect::<Vec<_>>())
            .unwrap(),
        vec![Some("Joseph".to_string()), None]
    );
    let ages = result[0]
        .with_dyn(|a| a.as_struct_array_unchecked().field(1))
        .unwrap();
    assert_eq!(
        ages.into_primitive().unwrap().maybe_null_slice::<i32>(),
        vec![25, 31]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_simple() {
    let expected_numbers_split: Vec<Vec<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(expected_numbers_split.iter().cloned().map(ArrayData::from))
            .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let writer = VortexFileWriter::new(Vec::new())
        .write_array_columns(expected_array.into_array())
        .await
        .unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    // test no indices
    let empty_indices = Vec::<u32>::new();
    let actual_kept_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from(empty_indices))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0_usize, 3, 99, 100, 101, 399, 400, 401, 499];
    let kept_indices_u16 = kept_indices.iter().map(|&x| x as u16).collect::<Vec<_>>();

    let actual_kept_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from(kept_indices_u16))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();
    let actual_kept_numbers_array = actual_kept_array
        .field(0)
        .unwrap()
        .into_primitive()
        .unwrap();

    let expected_kept_numbers: Vec<i16> =
        kept_indices.iter().map(|&x| expected_numbers[x]).collect();
    let actual_kept_numbers = actual_kept_numbers_array.maybe_null_slice::<i16>();

    assert_eq!(expected_kept_numbers, actual_kept_numbers);

    // test all indices
    let actual_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from((0u32..500).collect_vec()))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();
    let actual_numbers_array = actual_array.field(0).unwrap().into_primitive().unwrap();
    let actual_numbers = actual_numbers_array.maybe_null_slice::<i16>();

    assert_eq!(expected_numbers, actual_numbers);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_on_two_columns() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        PrimitiveArray::from(numbers_expected[..4].to_vec()).into_array(),
        PrimitiveArray::from(numbers_expected[4..].to_vec()).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    let kept_indices = [0_usize, 3, 7];
    let kept_indices_u8 = kept_indices.iter().map(|&x| x as u8).collect::<Vec<_>>();

    let array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from(kept_indices_u8))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();

    let strings_actual = array
        .field(0)
        .unwrap()
        .into_varbinview()
        .unwrap()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(
        strings_actual,
        kept_indices
            .iter()
            .map(|&x| strings_expected[x])
            .collect::<Vec<_>>()
    );

    let numbers_actual_array = array.field(1).unwrap().into_primitive().unwrap();
    let numbers_actual = numbers_actual_array.maybe_null_slice::<u32>();
    assert_eq!(
        numbers_actual,
        kept_indices
            .iter()
            .map(|&x| numbers_expected[x])
            .collect::<Vec<u32>>()
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_and_with_row_filter_simple() {
    let expected_numbers_split: Vec<Vec<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(expected_numbers_split.iter().cloned().map(ArrayData::from))
            .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let writer = VortexFileWriter::new(Vec::new())
        .write_array_columns(expected_array.into_array())
        .await
        .unwrap();
    let written = Buffer::from(writer.finalize().await.unwrap());

    // test no indices
    let empty_indices = Vec::<u32>::new();
    let actual_kept_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from(empty_indices))
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            Column::new_expr(Field::from("numbers")),
            Operator::Gt,
            Literal::new_expr(50_i16.into()),
        )))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0_usize, 3, 99, 100, 101, 399, 400, 401, 499];
    let kept_indices_u16 = kept_indices.iter().map(|&x| x as u16).collect::<Vec<_>>();

    let actual_kept_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from(kept_indices_u16))
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            Column::new_expr(Field::from("numbers")),
            Operator::Gt,
            Literal::new_expr(50_i16.into()),
        )))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();
    let actual_kept_numbers_array = actual_kept_array
        .field(0)
        .unwrap()
        .into_primitive()
        .unwrap();

    let expected_kept_numbers: Vec<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x])
        .filter(|&x| x > 50)
        .collect();
    let actual_kept_numbers = actual_kept_numbers_array.maybe_null_slice::<i16>();

    assert_eq!(expected_kept_numbers, actual_kept_numbers);

    // test all indices
    let actual_array = VortexReadBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_indices(ArrayData::from((0..500).collect_vec()))
        .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
            Column::new_expr(Field::from("numbers")),
            Operator::Gt,
            Literal::new_expr(50_i16.into()),
        )))
        .build()
        .await
        .unwrap()
        .read_all()
        .await
        .unwrap()
        .into_struct()
        .unwrap();
    let actual_numbers_array = actual_array.field(0).unwrap().into_primitive().unwrap();
    let actual_numbers = actual_numbers_array.maybe_null_slice::<i16>();

    assert_eq!(
        expected_numbers
            .iter()
            .filter(|&&x| x > 50)
            .cloned()
            .collect::<Vec<_>>(),
        actual_numbers
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string_chunked() {
    let name_chunk1 = ArrayData::from_iter(vec![
        Some("Joseph".to_owned()),
        Some("James".to_owned()),
        Some("Angela".to_owned()),
    ]);
    let age_chunk1 = ArrayData::from_iter(vec![Some(25_i32), Some(31), None]);
    let name_chunk2 = ArrayData::from_iter(vec![
        Some("Pharrell".to_owned()),
        Some("Khalil".to_owned()),
        Some("Mikhail".to_owned()),
        None,
    ]);
    let age_chunk2 = ArrayData::from_iter(vec![Some(57_i32), Some(18), None, Some(32)]);

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

    let buffer = Vec::new();
    let written_bytes = VortexFileWriter::new(buffer)
        .write_array_columns(array)
        .await
        .unwrap()
        .finalize()
        .await
        .unwrap();
    let actual_array =
        VortexReadBuilder::new(Buffer::from(written_bytes), LayoutDeserializer::default())
            .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
                Column::new_expr(Field::from("name")),
                Operator::Eq,
                Literal::new_expr("Joseph".into()),
            )))
            .build()
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();

    assert_eq!(actual_array.len(), 1);
    let names = actual_array
        .with_dyn(|a| a.as_struct_array_unchecked().field(0))
        .unwrap();
    assert_eq!(
        names
            .into_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = actual_array
        .with_dyn(|a| a.as_struct_array_unchecked().field(1))
        .unwrap();
    assert_eq!(
        ages.into_primitive().unwrap().maybe_null_slice::<i32>(),
        vec![25]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_pruning_with_or() {
    let letter_chunk1 = ArrayData::from_iter(vec![
        Some("A".to_owned()),
        Some("B".to_owned()),
        Some("D".to_owned()),
    ]);
    let number_chunk1 = ArrayData::from_iter(vec![Some(25_i32), Some(31), None]);
    let letter_chunk2 = ArrayData::from_iter(vec![
        Some("G".to_owned()),
        Some("I".to_owned()),
        Some("J".to_owned()),
        None,
    ]);
    let number_chunk2 = ArrayData::from_iter(vec![Some(4_i32), Some(18), None, Some(21)]);
    let letter_chunk3 = ArrayData::from_iter(vec![
        Some("L".to_owned()),
        None,
        Some("O".to_owned()),
        Some("P".to_owned()),
    ]);
    let number_chunk3 = ArrayData::from_iter(vec![Some(10_i32), Some(15), None, Some(22)]);
    let letter_chunk4 = ArrayData::from_iter(vec![
        Some("X".to_owned()),
        Some("Y".to_owned()),
        Some("Z".to_owned()),
    ]);
    let number_chunk4 = ArrayData::from_iter(vec![Some(66_i32), Some(77), Some(88)]);

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

    let buffer = Vec::new();
    let written_bytes = VortexFileWriter::new(buffer)
        .write_array_columns(array)
        .await
        .unwrap()
        .finalize()
        .await
        .unwrap();
    let actual_array =
        VortexReadBuilder::new(Buffer::from(written_bytes), LayoutDeserializer::default())
            .with_row_filter(RowFilter::new(BinaryExpr::new_expr(
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("letter")),
                    Operator::Lte,
                    Literal::new_expr("J".into()),
                ),
                Operator::Or,
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("number")),
                    Operator::Lt,
                    Literal::new_expr(25.into()),
                ),
            )))
            .build()
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();

    assert_eq!(actual_array.len(), 10);
    let letters = actual_array
        .with_dyn(|a| a.as_struct_array_unchecked().field(0))
        .unwrap();
    assert_eq!(
        letters
            .into_varbinview()
            .unwrap()
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
    let numbers = actual_array
        .with_dyn(|a| a.as_struct_array_unchecked().field(1))
        .unwrap();
    assert_eq!(
        (0..numbers.len())
            .map(|index| -> Option<i32> {
                scalar_at(&numbers, index)
                    .unwrap()
                    .as_primitive()
                    .typed_value::<i32>()
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
