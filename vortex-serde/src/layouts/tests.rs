#![allow(clippy::panic)]

use std::iter;
use std::sync::Arc;

use futures::StreamExt;
use vortex::accessor::ArrayAccessor;
use vortex::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex::validity::Validity;
use vortex::variants::StructArrayTrait;
use vortex::{ArrayDType, IntoArray, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_expr::{BinaryExpr, Column, Literal, Operator};

use crate::layouts::write::LayoutWriter;
use crate::layouts::{
    LayoutDeserializer, LayoutReaderBuilder, Projection, RowFilter, CHUNKED_LAYOUT_ID,
    COLUMN_LAYOUT_ID, EOF_SIZE, FLAT_LAYOUT_ID, FOOTER_POSTSCRIPT_SIZE, INLINE_SCHEMA_LAYOUT_ID,
    MAGIC_BYTES, VERSION,
};

#[test]
fn test_constants() {
    // the footer postscript size can change iff we increment the version
    // i.e., it must be 32 bytes iff VERSION == 1
    assert_eq!(VERSION, 1);
    assert_eq!(FOOTER_POSTSCRIPT_SIZE, 32);

    // these constants can never change (without breaking all existing files)
    assert_eq!(MAGIC_BYTES, *b"VRTX");
    assert_eq!(EOF_SIZE, 8);
    assert_eq!(FLAT_LAYOUT_ID.0, 1);
    assert_eq!(CHUNKED_LAYOUT_ID.0, 2);
    assert_eq!(COLUMN_LAYOUT_ID.0, 3);
    assert_eq!(INLINE_SCHEMA_LAYOUT_ID.0, 4);
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
    let mut writer = LayoutWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = writer.finalize().await.unwrap();

    let mut stream = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
        .with_batch_size(5)
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
    let mut writer = LayoutWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = writer.finalize().await.unwrap();

    let array = LayoutReaderBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::new([0]))
        .with_batch_size(5)
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

    let array = LayoutReaderBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::Flat(vec![Field::Name("strings".to_string())]))
        .with_batch_size(5)
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

    let array = LayoutReaderBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::new([1]))
        .with_batch_size(5)
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

    let array = LayoutReaderBuilder::new(written.clone(), LayoutDeserializer::default())
        .with_projection(Projection::Flat(vec![Field::Name("numbers".to_string())]))
        .with_batch_size(5)
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
    let mut writer = LayoutWriter::new(buf);
    writer = writer.write_array_columns(st.into_array()).await.unwrap();
    let written = writer.finalize().await.unwrap();

    let mut stream = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
        .with_batch_size(5)
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
            panic!("Expected column doesn't exist")
        }
    }
    assert_eq!(item_count, 10);
    assert_eq!(batch_count, 2);
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
    let mut writer = LayoutWriter::new(buf);
    writer = writer.write_array_columns(chunked_st).await.unwrap();
    let written = writer.finalize().await.unwrap();
    let mut reader = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
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
    let mut writer = LayoutWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();
    let written = writer.finalize().await.unwrap();
    let mut reader = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("name"))),
            Operator::Eq,
            Arc::new(Literal::new("Joseph".into())),
        ))))
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
        vec![names.clone().into_array(), ages.clone().into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut writer = LayoutWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();
    let written = writer.finalize().await.unwrap();
    let mut reader = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("name"))),
                Operator::Eq,
                Arc::new(Literal::new("Angela".into())),
            )),
            Operator::Or,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new(Field::from("age"))),
                    Operator::Gte,
                    Arc::new(Literal::new(20.into())),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new(Field::from("age"))),
                    Operator::Lte,
                    Arc::new(Literal::new(30.into())),
                )),
            )),
        ))))
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
async fn filter_and() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_nullable_vec(vec![Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names.clone().into_array(), ages.clone().into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let mut writer = LayoutWriter::new(Vec::new());
    writer = writer.write_array_columns(st).await.unwrap();
    let written = writer.finalize().await.unwrap();
    let mut reader = LayoutReaderBuilder::new(written, LayoutDeserializer::default())
        .with_row_filter(RowFilter::new(Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("age"))),
                Operator::Gt,
                Arc::new(Literal::new(21.into())),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("age"))),
                Operator::Lte,
                Arc::new(Literal::new(33.into())),
            )),
        ))))
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
