#![no_main]

use arrow_buffer::BooleanBuffer;
use arrow_ord::ord::make_comparator;
use arrow_ord::sort::SortOptions;
use futures_util::TryStreamExt;
use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::compute::{Operator, compare};
use vortex_array::stream::ArrayStreamArrayExt;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::ByteBufferMut;
use vortex_dtype::{DType, StructDType};
use vortex_error::VortexUnwrap;
use vortex_file::{VortexOpenOptions, VortexWriteOptions};

fuzz_target!(|array_data: ArbitraryArray| -> Corpus {
    let array_data = array_data.0;

    if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype()) {
        return Corpus::Reject;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .vortex_unwrap();

    runtime.block_on(async move {
        let full_buff = VortexWriteOptions::default()
            .write(ByteBufferMut::empty(), array_data.to_array_stream())
            .await
            .vortex_unwrap();

        let output = VortexOpenOptions::in_memory(full_buff)
            .open()
            .await
            .vortex_unwrap()
            .scan()
            .into_array_stream()
            .vortex_unwrap()
            .try_collect::<Vec<_>>()
            .await
            .vortex_unwrap();

        let output = if output.is_empty() {
            ChunkedArray::try_new(output, array_data.dtype().clone())
                .vortex_unwrap()
                .into_array()
        } else {
            ChunkedArray::from_iter(output).into_array()
        };

        assert_eq!(array_data.len(), output.len(), "Length was not preserved.");
        assert_eq!(
            array_data.dtype(),
            output.dtype(),
            "DTypes aren't preserved expected {}, actual {}",
            array_data.dtype(),
            output.dtype()
        );

        if matches!(array_data.dtype(), DType::Struct(_, _) | DType::List(_, _)) {
            compare_struct(array_data, output);
        } else {
            let r = compare(&array_data, &output, Operator::Eq).vortex_unwrap();
            let true_count = r
                .to_bool()
                .vortex_unwrap()
                .boolean_buffer()
                .count_set_bits();
            assert_eq!(true_count, array_data.len());
        }
    });

    Corpus::Keep
});

fn compare_struct(expected: ArrayRef, actual: ArrayRef) {
    let arrow_expected = expected.clone().into_arrow_preferred().vortex_unwrap();
    let arrow_actual = actual.clone().into_arrow_preferred().vortex_unwrap();

    let cmp_fn =
        make_comparator(&arrow_expected, &arrow_actual, SortOptions::default()).vortex_unwrap();

    let comparison_result =
        BooleanBuffer::collect_bool(arrow_expected.len(), |idx| cmp_fn(idx, idx).is_eq());

    assert_eq!(
        comparison_result.count_set_bits(),
        arrow_expected.len(),
        "\nEXPECTED: {}ACTUAL: {}",
        expected.tree_display(),
        actual.tree_display()
    );
}

fn has_nullable_struct(dtype: &DType) -> bool {
    dtype.is_struct() && dtype.is_nullable()
        || dtype
            .as_struct()
            .map(|sdt| sdt.fields().any(|dtype| has_nullable_struct(&dtype)))
            .unwrap_or(false)
        || dtype
            .as_list_element()
            .map(has_nullable_struct)
            .unwrap_or(false)
}

fn has_duplicate_field_names(dtype: &DType) -> bool {
    if let Some(struct_dtype) = dtype.as_struct() {
        struct_has_duplicate_names(struct_dtype)
    } else if let Some(list_elem) = dtype.as_list_element() {
        has_duplicate_field_names(list_elem)
    } else {
        false
    }
}

fn struct_has_duplicate_names(struct_dtype: &StructDType) -> bool {
    HashSet::from_iter(struct_dtype.names().iter()).len() != struct_dtype.names().len()
        || struct_dtype
            .fields()
            .any(|dtype| has_duplicate_field_names(&dtype))
}
