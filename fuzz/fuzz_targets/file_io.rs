#![no_main]

use arrow_buffer::BooleanBufferBuilder;
use arrow_ord::ord::make_comparator;
use arrow_ord::sort::SortOptions;
use bytes::Bytes;
use futures_util::TryStreamExt;
use libfuzzer_sys::{fuzz_target, Corpus};
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::compute::{compare, Operator};
use vortex_array::stream::ArrayStreamArrayExt;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_file::{VortexOpenOptions, VortexWriteOptions};

fuzz_target!(|array_data: ArbitraryArray| -> Corpus {
    let array_data = array_data.0;

    if !array_data.dtype().is_struct() {
        return Corpus::Reject;
    }

    if has_nullable_struct(array_data.dtype()) {
        return Corpus::Reject;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async move {
        let buf = Vec::new();
        let full_buff = VortexWriteOptions::default()
            .write(buf, array_data.to_array_stream())
            .await
            .unwrap();

        let written = Bytes::from(full_buff);

        let output = VortexOpenOptions::in_memory(written)
            .open()
            .await
            .unwrap()
            .scan()
            .into_array_stream()
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
            let r = compare(&array_data, &output, Operator::Eq).unwrap();
            let true_count = r.to_bool().unwrap().boolean_buffer().count_set_bits();
            assert_eq!(true_count, array_data.len());
        }
    });

    Corpus::Keep
});

fn compare_struct(expected: ArrayRef, actual: ArrayRef) {
    assert_eq!(
        expected.dtype(),
        actual.dtype(),
        "DTypes aren't preserved expected {}, actual {}",
        expected.dtype(),
        actual.dtype()
    );
    assert_eq!(
        expected.len(),
        actual.len(),
        "Arrays length isn't preserved expected: {}, actual: {}",
        expected.len(),
        actual.len()
    );

    if expected.dtype().as_struct().unwrap().names().is_empty()
        && actual.dtype().as_struct().unwrap().names().is_empty()
    {
        return;
    }

    let arrow_lhs = expected.clone().into_arrow_preferred().unwrap();
    let arrow_rhs = actual.clone().into_arrow_preferred().unwrap();

    let cmp_fn = make_comparator(&arrow_lhs, &arrow_rhs, SortOptions::default()).unwrap();

    let mut bool_builder = BooleanBufferBuilder::new(arrow_lhs.len());

    for idx in 0..arrow_lhs.len() {
        bool_builder.append(cmp_fn(idx, idx).is_eq());
    }

    assert_eq!(
        bool_builder.finish().count_set_bits(),
        arrow_lhs.len(),
        "\nEXPECTED: {}ACTUAL: {}",
        expected.tree_display(),
        actual.tree_display()
    );
}

fn has_nullable_struct(dtype: &DType) -> bool {
    dtype.is_nullable()
        || dtype
            .as_struct()
            .map(|sdt| sdt.fields().any(|dtype| has_nullable_struct(&dtype)))
            .unwrap_or(false)
}
