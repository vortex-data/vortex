#![feature(float_next_up_down)]

use std::process::ExitCode;
use std::sync::Arc;

use prettytable::{Cell, Row, Table};
use vortex::array::builder::VarBinBuilder;
use vortex::array::{
    BoolArray, ChunkedArray, ConstantArray, NullArray, PrimitiveArray, SparseArray, StructArray,
    VarBinViewArray,
};
use vortex::compute::Operator;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};
use vortex_alp::{ALPArray, Exponents, RDEncoder};
use vortex_bytebool::ByteBoolArray;
use vortex_datetime_dtype::{TemporalMetadata, TimeUnit, TIME_ID};
use vortex_datetime_parts::DateTimePartsArray;
use vortex_dict::DictArray;
use vortex_dtype::{DType, ExtDType, Nullability, PType};
use vortex_fastlanes::{BitPackedArray, DeltaArray, FoRArray};
use vortex_fsst::{fsst_compress, fsst_train_compressor};
use vortex_roaring::{Bitmap, RoaringBoolArray, RoaringIntArray};
use vortex_runend::RunEndArray;
use vortex_runend_bool::RunEndBoolArray;
use vortex_scalar::ScalarValue;
use vortex_zigzag::ZigZagArray;

const OPERATORS: [Operator; 6] = [
    Operator::Lte,
    Operator::Lt,
    Operator::Gt,
    Operator::Gte,
    Operator::Eq,
    Operator::NotEq,
];

fn fsst_array() -> Array {
    let input_array = varbin_array();
    let compressor = fsst_train_compressor(&input_array).unwrap();

    fsst_compress(&input_array, &compressor)
        .unwrap()
        .into_array()
}

fn varbin_array() -> Array {
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

fn varbinview_array() -> Array {
    VarBinViewArray::from_iter_str(vec![
        "The Greeks never said that the limit could not be overstepped",
        "They said it existed and that whoever dared to exceed it was mercilessly struck down",
        "Nothing in present history can contradict them",
    ])
    .into_array()
}

fn enc_impls() -> Vec<Array> {
    vec![
        ALPArray::try_new(
            PrimitiveArray::from(vec![1]).into_array(),
            Exponents { e: 1, f: 1 },
            None,
        )
        .unwrap()
        .into_array(),
        RDEncoder::new(&[1.123_848_f32.powi(-2)])
            .encode(&PrimitiveArray::from(vec![0.1f64.next_up()]))
            .into_array(),
        BitPackedArray::encode(&PrimitiveArray::from(vec![100u32]).into_array(), 8)
            .unwrap()
            .into_array(),
        BoolArray::from(vec![false]).into_array(),
        ByteBoolArray::from(vec![false]).into_array(),
        ChunkedArray::try_new(
            vec![
                BoolArray::from(vec![false]).into_array(),
                BoolArray::from(vec![true]).into_array(),
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
            PrimitiveArray::from(vec![1]).into_array(),
            PrimitiveArray::from(vec![0]).into_array(),
            PrimitiveArray::from(vec![0]).into_array(),
        )
        .unwrap()
        .into_array(),
        DeltaArray::try_from_primitive_array(&PrimitiveArray::from(vec![0u32, 1]))
            .unwrap()
            .into_array(),
        DictArray::try_new(
            PrimitiveArray::from(vec![0u32, 1, 0]).into_array(),
            PrimitiveArray::from(vec![1, 2]).into_array(),
        )
        .unwrap()
        .into_array(),
        fsst_array(),
        FoRArray::try_new(
            PrimitiveArray::from(vec![0u32, 1, 2]).into_array(),
            10.into(),
            5,
        )
        .unwrap()
        .into_array(),
        NullArray::new(10).into_array(),
        PrimitiveArray::from(vec![0, 1]).into_array(),
        RoaringBoolArray::try_new(Bitmap::from([0u32, 10, 20]), 30)
            .unwrap()
            .into_array(),
        RoaringIntArray::try_new(Bitmap::from([5u32, 6, 8]), PType::U32)
            .unwrap()
            .into_array(),
        RunEndArray::try_new(
            PrimitiveArray::from(vec![5u32, 8]).into_array(),
            PrimitiveArray::from(vec![0, 1]).into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
        RunEndBoolArray::try_new(
            PrimitiveArray::from(vec![5u32, 8]).into_array(),
            true,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
        SparseArray::try_new(
            PrimitiveArray::from(vec![5u64, 8]).into_array(),
            PrimitiveArray::from_vec(vec![3u32, 6], Validity::AllValid).into_array(),
            10,
            ScalarValue::Null,
        )
        .unwrap()
        .into_array(),
        StructArray::try_new(
            ["a".into(), "b".into()].into(),
            vec![
                PrimitiveArray::from(vec![0, 1, 2]).into_array(),
                PrimitiveArray::from(vec![0.1f64, 1.1f64, 2.1f64]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
        varbin_array(),
        varbinview_array(),
        ZigZagArray::encode(&PrimitiveArray::from(vec![-1, 1, -9, 9]).into_array()).unwrap(),
    ]
}

fn bool_to_cell(val: bool) -> Cell {
    let style = if val { "bcFdBG" } else { "bcBr" };
    Cell::new(if val { "✓" } else { "𐄂" }).style_spec(style)
}

fn compute_funcs(encodings: &[Array]) {
    let mut table = Table::new();
    table.add_row(Row::new(
        vec![
            "Encoding",
            "cast",
            "fill_forward",
            "filter",
            "scalar_at",
            "subtract_scalar",
            "search_sorted",
            "slice",
            "take",
            "and",
            "or",
        ]
        .into_iter()
        .map(Cell::new)
        .collect(),
    ));
    for arr in encodings {
        let mut impls = vec![Cell::new(arr.encoding().id().as_ref())];
        impls.push(bool_to_cell(arr.with_dyn(|a| a.cast().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.fill_forward().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.filter().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.scalar_at().is_some())));
        impls.push(bool_to_cell(
            arr.with_dyn(|a| a.subtract_scalar().is_some()),
        ));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.search_sorted().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.slice().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.take().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.and().is_some())));
        impls.push(bool_to_cell(arr.with_dyn(|a| a.or().is_some())));
        table.add_row(Row::new(impls));
    }
    table.printstd();
}

fn compare_funcs(encodings: &[Array]) {
    for arr in encodings {
        println!("\nArray {} compare functions", arr.encoding().id().as_ref());
        let mut table = Table::new();
        table.add_row(Row::new(
            [Cell::new("Encoding")]
                .into_iter()
                .chain(OPERATORS.iter().map(|a| Cell::new(a.to_string().as_ref())))
                .collect(),
        ));
        for arr2 in encodings {
            let mut impls = vec![Cell::new(arr2.encoding().id().as_ref())];
            for op in OPERATORS {
                impls.push(bool_to_cell(
                    arr.with_dyn(|a1| a1.compare(arr2, op).is_some()),
                ));
            }
            table.add_row(Row::new(impls));
        }
        table.printstd();
    }
}

fn main() -> ExitCode {
    let arrays = enc_impls();
    compute_funcs(&arrays);
    compare_funcs(&arrays);
    ExitCode::SUCCESS
}
