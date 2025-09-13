// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![allow(clippy::result_large_err)]

use arrow_buffer::BooleanBuffer;
use arrow_ord::ord::make_comparator;
use arrow_ord::sort::SortOptions;
use itertools::Itertools;
use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::compute::{Operator, compare, filter};
use vortex_array::{Array, ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_buffer::ByteBufferMut;
use vortex_dtype::{DType, StructFields};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
use vortex_expr::{Scope, lit, root};
use vortex_file::{VortexOpenOptions, VortexWriteOptions};
use vortex_fuzz::FuzzFileAction;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

fuzz_target!(|fuzz: FuzzFileAction| -> Corpus {
    let FuzzFileAction {
        array,
        projection_expr,
        filter_expr,
    } = fuzz;
    let array_data = array;

    if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype()) {
        return Corpus::Reject;
    }

    let expected_array = {
        let bool_mask = filter_expr
            .clone()
            .unwrap_or_else(|| lit(true))
            .evaluate(&Scope::new(array_data.clone()))
            .vortex_unwrap();
        let mask = bool_mask.to_bool().to_mask_fill_null_false();
        let filtered = filter(&array_data, &mask).vortex_unwrap();
        projection_expr
            .clone()
            .unwrap_or_else(|| root())
            .evaluate(&Scope::new(filtered))
            .vortex_unwrap()
    };

    let mut full_buff = ByteBufferMut::empty();
    let _footer = VortexWriteOptions::default()
        .blocking::<SingleThreadRuntime>()
        .write(&mut full_buff, array_data.to_array_iterator())
        .vortex_unwrap();

    let mut output = VortexOpenOptions::new()
        .open_buffer(full_buff)
        .vortex_unwrap()
        .scan()
        .vortex_unwrap()
        .with_projection(projection_expr.unwrap_or_else(|| root()))
        .with_some_filter(filter_expr)
        .into_array_iter()
        .vortex_unwrap()
        .try_collect::<_, Vec<_>, _>()
        .vortex_unwrap();

    let output_array = match output.len() {
        0 => Canonical::empty(expected_array.dtype()).into_array(),
        1 => output.pop().vortex_expect("one output"),
        _ => ChunkedArray::from_iter(output).into_array(),
    };

    assert_eq!(
        expected_array.len(),
        output_array.len(),
        "Length was not preserved."
    );
    assert_eq!(
        expected_array.dtype(),
        output_array.dtype(),
        "DTypes aren't preserved expected {}, actual {}",
        expected_array.dtype(),
        output_array.dtype()
    );

    if matches!(
        expected_array.dtype(),
        DType::Struct(_, _) | DType::List(_, _)
    ) {
        compare_struct(expected_array, output_array);
    } else {
        let bool_result = compare(&expected_array, &output_array, Operator::Eq)
            .vortex_unwrap()
            .to_bool();
        let true_count = bool_result.boolean_buffer().count_set_bits();
        if true_count != expected_array.len()
            && (bool_result.all_valid() || expected_array.all_valid())
        {
            vortex_panic!(
                "Failed to match original array {}with{}",
                expected_array.display_tree(),
                output_array.display_tree()
            );
        }
    }

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
        expected.display_tree(),
        actual.display_tree()
    );
}

fn has_nullable_struct(dtype: &DType) -> bool {
    dtype.is_struct() && dtype.is_nullable()
        || dtype
            .as_struct_fields_opt()
            .map(|sdt| sdt.fields().any(|dtype| has_nullable_struct(&dtype)))
            .unwrap_or(false)
        || dtype
            .as_list_element_opt()
            .map(|e| has_nullable_struct(e.as_ref()))
            .unwrap_or(false)
}

fn has_duplicate_field_names(dtype: &DType) -> bool {
    if let Some(struct_dtype) = dtype.as_struct_fields_opt() {
        struct_has_duplicate_names(struct_dtype)
    } else if let Some(list_elem) = dtype.as_list_element_opt() {
        has_duplicate_field_names(list_elem)
    } else {
        false
    }
}

fn struct_has_duplicate_names(struct_dtype: &StructFields) -> bool {
    HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
        != struct_dtype.names().len()
        || struct_dtype
            .fields()
            .any(|dtype| has_duplicate_field_names(&dtype))
}
