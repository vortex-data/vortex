#![no_main]

use arrow_buffer::BooleanBufferBuilder;
use arrow_ord::ord::make_comparator;
use arrow_ord::sort::SortOptions;
use bytes::Bytes;
use futures_util::TryStreamExt;
use libfuzzer_sys::{fuzz_target, Corpus};
use vortex_array::array::ChunkedArray;
use vortex_array::compute::{compare, Operator};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant, IntoCanonical};
use vortex_file::{Scan, VortexOpenOptions, VortexWriteOptions};
use vortex_sampling_compressor::ALL_ENCODINGS_CONTEXT;

fuzz_target!(|array_data: ArrayData| -> Corpus {
    if !array_data.dtype().is_struct() {
        return Corpus::Reject;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async move {
        let buf = Vec::new();
        let full_buff = VortexWriteOptions::default()
            .write(buf, array_data.clone().into_array_stream())
            .await
            .unwrap();

        let written = Bytes::from(full_buff);

        let output = VortexOpenOptions::new(ALL_ENCODINGS_CONTEXT.clone())
            .open(written)
            .await
            .unwrap()
            .scan(Scan::all())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let output = if output.is_empty() {
            ChunkedArray::try_new(output, array_data.dtype().clone())
                .unwrap()
                .into_array()
        } else {
            ChunkedArray::from_iter(output).into_array()
        };

        assert_eq!(array_data.len(), output.len(), "Length was not preserved.");

        if array_data.dtype().is_struct() {
            compare_struct(array_data, output);
        } else {
            let r = compare(&array_data, output, Operator::Eq).unwrap();
            let true_count = r.into_bool().unwrap().boolean_buffer().count_set_bits();
            assert_eq!(true_count, array_data.len());
        }
    });

    Corpus::Keep
});

fn compare_struct(lhs: ArrayData, rhs: ArrayData) {
    assert!(lhs.dtype().eq_ignore_nullability(rhs.dtype()));
    assert_eq!(lhs.len(), rhs.len(), "Arrays length isn't preserved");

    if lhs.dtype().as_struct().unwrap().names().len() == 0
        && lhs.dtype().as_struct().unwrap().names().len()
            == rhs.dtype().as_struct().unwrap().names().len()
    {
        return;
    }

    if let Some(st) = lhs.dtype().as_struct() {
        if st.names().len() == 0 {
            return;
        }
    }

    let arrow_lhs = lhs.clone().into_arrow().unwrap();
    let arrow_rhs = rhs.clone().into_arrow().unwrap();

    let cmp_fn = make_comparator(&arrow_lhs, &arrow_rhs, SortOptions::default()).unwrap();

    let mut bool_builder = BooleanBufferBuilder::new(arrow_lhs.len());

    for idx in 0..arrow_lhs.len() {
        bool_builder.append(cmp_fn(idx, idx).is_eq());
    }

    assert_eq!(
        bool_builder.finish().count_set_bits(),
        arrow_lhs.len(),
        "\nLHS: {}RHS: {}",
        lhs.tree_display(),
        rhs.tree_display()
    );
}
