#![feature(float_next_up_down)]

use std::process::ExitCode;
use std::sync::Arc;

use tabled::builder::Builder;
use tabled::settings::themes::Colorization;
use tabled::settings::{Color, Style};
use vortex::array::builder::VarBinBuilder;
use vortex::array::{
    BoolArray, ChunkedArray, ConstantArray, ListArray, NullArray, PrimitiveArray, SparseArray,
    StructArray, VarBinViewArray,
};
use vortex::buffer::buffer;
use vortex::datetime_dtype::{TemporalMetadata, TimeUnit, TIME_ID};
use vortex::dtype::{DType, ExtDType, Nullability, PType};
use vortex::encodings::alp::{ALPArray, Exponents, RDEncoder};
use vortex::encodings::bytebool::ByteBoolArray;
use vortex::encodings::datetime_parts::DateTimePartsArray;
use vortex::encodings::dict::DictArray;
use vortex::encodings::fastlanes::{BitPackedArray, DeltaArray, FoRArray};
use vortex::encodings::fsst::{fsst_compress, fsst_train_compressor};
use vortex::encodings::roaring::{Bitmap, RoaringBoolArray, RoaringIntArray};
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend_bool::RunEndBoolArray;
use vortex::encodings::zigzag::ZigZagArray;
use vortex::scalar::Scalar;
use vortex::validity::Validity;
use vortex::{ArrayData, IntoArrayData};

fn fsst_array() -> ArrayData {
    let input_array = varbin_array();
    let compressor = fsst_train_compressor(&input_array).unwrap();

    fsst_compress(&input_array, &compressor)
        .unwrap()
        .into_array()
}

fn varbin_array() -> ArrayData {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.push_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.push_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.push_value(b"Nothing in present history can contradict them");
    input_array
        .finish(DType::Utf8(Nullability::NonNullable))
        .into_array()
}

fn varbinview_array() -> ArrayData {
    VarBinViewArray::from_iter_str(vec![
        "The Greeks never said that the limit could not be overstepped",
        "They said it existed and that whoever dared to exceed it was mercilessly struck down",
        "Nothing in present history can contradict them",
    ])
    .into_array()
}

fn enc_impls() -> Vec<ArrayData> {
    vec![
        ALPArray::try_new(buffer![1].into_array(), Exponents { e: 1, f: 1 }, None)
            .unwrap()
            .into_array(),
        RDEncoder::new(&[1.123_848_f32.powi(-2)])
            .encode(&PrimitiveArray::new(
                buffer![0.1f64.next_up()],
                Validity::NonNullable,
            ))
            .into_array(),
        BitPackedArray::encode(&buffer![100u32].into_array(), 8)
            .unwrap()
            .into_array(),
        BoolArray::from_iter([false]).into_array(),
        ByteBoolArray::from(vec![false]).into_array(),
        ChunkedArray::try_new(
            vec![
                BoolArray::from_iter([false]).into_array(),
                BoolArray::from_iter([true]).into_array(),
            ],
            DType::Bool(Nullability::NonNullable),
        )
        .unwrap()
        .into_array(),
        ConstantArray::new(10, 1).into_array(),
        DateTimePartsArray::try_new(
            DType::Extension(Arc::new(ExtDType::new(
                TIME_ID.clone(),
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Some(TemporalMetadata::Time(TimeUnit::S).into()),
            ))),
            buffer![1].into_array(),
            buffer![0].into_array(),
            buffer![0].into_array(),
        )
        .unwrap()
        .into_array(),
        DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
            buffer![0u32, 1],
            Validity::NonNullable,
        ))
        .unwrap()
        .into_array(),
        DictArray::try_new(buffer![0u32, 1, 0].into_array(), buffer![1, 2].into_array())
            .unwrap()
            .into_array(),
        fsst_array(),
        FoRArray::try_new(buffer![0u32, 1, 2].into_array(), 10.into(), 5)
            .unwrap()
            .into_array(),
        ListArray::try_new(
            buffer![0, 1].into_array(),
            buffer![0, 1, 2].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
        NullArray::new(10).into_array(),
        buffer![0, 1].into_array(),
        RoaringBoolArray::try_new(Bitmap::from([0u32, 10, 20]), 30)
            .unwrap()
            .into_array(),
        RoaringIntArray::try_new(Bitmap::from([5u32, 6, 8]), PType::U32)
            .unwrap()
            .into_array(),
        RunEndArray::try_new(buffer![5u32, 8].into_array(), buffer![0, 1].into_array())
            .unwrap()
            .into_array(),
        RunEndBoolArray::try_new(buffer![5u32, 8].into_array(), true, Validity::NonNullable)
            .unwrap()
            .into_array(),
        SparseArray::try_new(
            buffer![5u64, 8].into_array(),
            PrimitiveArray::new(buffer![3u32, 6], Validity::AllValid).into_array(),
            10,
            Scalar::null_typed::<u32>(),
        )
        .unwrap()
        .into_array(),
        StructArray::try_new(
            ["a".into(), "b".into()].into(),
            vec![
                buffer![0, 1, 2].into_array(),
                buffer![0.1f64, 1.1f64, 2.1f64].into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
        varbin_array(),
        varbinview_array(),
        ZigZagArray::encode(&buffer![-1, 1, -9, 9].into_array())
            .unwrap()
            .into_array(),
    ]
}

fn compute_funcs(encodings: &[ArrayData]) {
    let mut table_builder = Builder::default();
    table_builder.push_record(vec![
        "Encoding",
        "cast",
        "compare",
        "fill_forward",
        "fill_null",
        "filter",
        "scalar_at",
        "binary_numeric",
        "search_sorted",
        "slice",
        "take",
    ]);
    let mut colours = Vec::new();
    for (i, arr) in encodings.iter().enumerate() {
        let encoding = arr.encoding();
        let id = encoding.id();
        let mut impls = vec![id.as_ref()];
        for (j, func) in [
            encoding.cast_fn().is_some(),
            encoding.compare_fn().is_some(),
            encoding.fill_forward_fn().is_some(),
            encoding.fill_null_fn().is_some(),
            encoding.filter_fn().is_some(),
            encoding.scalar_at_fn().is_some(),
            encoding.binary_numeric_fn().is_some(),
            encoding.search_sorted_fn().is_some(),
            encoding.slice_fn().is_some(),
            encoding.take_fn().is_some(),
        ]
        .into_iter()
        .enumerate()
        {
            impls.push(if func { "✓" } else { "𐄂" });
            colours.push(Colorization::exact(
                [if func {
                    Color::BG_BRIGHT_GREEN
                } else {
                    Color::BG_RED
                }],
                (i + 1, j + 1),
            ));
        }
        table_builder.push_record(impls);
    }
    let mut table = table_builder.build();
    table.with(Style::modern());

    for color in colours.into_iter() {
        table.with(color);
    }

    println!("{table}");
}

fn main() -> ExitCode {
    let arrays = enc_impls();
    compute_funcs(&arrays);
    ExitCode::SUCCESS
}
