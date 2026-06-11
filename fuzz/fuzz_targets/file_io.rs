// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use itertools::Itertools;
use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_fuzz::CompressorStrategy;
use vortex_fuzz::FuzzFileAction;
use vortex_fuzz::RUNTIME;
use vortex_fuzz::SESSION;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

fuzz_target!(|fuzz: FuzzFileAction| -> Corpus {
    let FuzzFileAction {
        array,
        projection_expr,
        filter_expr,
        compressor_strategy,
        layout_reader_cache,
    } = fuzz;
    let array_data = array;

    if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype()) {
        return Corpus::Reject;
    }

    let mut ctx = SESSION.create_execution_ctx();

    // Baseline: evaluate the filter and projection eagerly in memory *before* writing the file.
    // The expression space includes fallible functions (cast, arithmetic, like), so a runtime
    // error here is not a bug; reject the input. A successful baseline is the oracle: once it
    // succeeds, the scan below must succeed and produce the same values.
    let Ok(bool_mask) = array_data
        .clone()
        .apply(&filter_expr.clone().unwrap_or_else(|| lit(true)))
        .and_then(|m| m.execute::<BoolArray>(&mut ctx))
    else {
        return Corpus::Reject;
    };
    let mask = bool_mask.to_mask_fill_null_false(&mut ctx);
    let filtered = array_data
        .filter(mask)
        .vortex_expect("filter operation should succeed in fuzz test");
    let Ok(expected_array) = filtered
        .apply(&projection_expr.clone().unwrap_or_else(root))
        .and_then(|a| a.execute::<Canonical>(&mut ctx))
        .map(Canonical::into_array)
    else {
        return Corpus::Reject;
    };

    let write_options = match compressor_strategy {
        CompressorStrategy::Default => SESSION.write_options(),
        CompressorStrategy::Compact => SESSION.write_options().with_strategy(
            WriteStrategyBuilder::default()
                .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
                .build(),
        ),
    };

    let mut full_buff = ByteBufferMut::empty();
    let _footer = write_options
        .blocking(&*RUNTIME)
        .write(&mut full_buff, array_data.to_array_iterator())
        .vortex_expect("file write should succeed in fuzz test");

    let open_options = if layout_reader_cache {
        SESSION.open_options().with_layout_reader_cache()
    } else {
        SESSION.open_options()
    };
    let file = open_options
        .open_buffer(full_buff)
        .vortex_expect("open_buffer should succeed in fuzz test");

    let run_scan = || {
        let mut output = file
            .scan()
            .vortex_expect("scan should succeed in fuzz test")
            .with_projection(projection_expr.clone().unwrap_or_else(root))
            .with_some_filter(filter_expr.clone())
            .into_array_iter(&*RUNTIME)
            .vortex_expect("into_array_iter should succeed in fuzz test")
            .try_collect::<_, Vec<_>, _>()
            .vortex_expect("collect should succeed in fuzz test");

        match output.len() {
            0 => Canonical::empty(expected_array.dtype()).into_array(),
            1 => output.pop().vortex_expect("one output"),
            _ => ChunkedArray::from_iter(output).into_array(),
        }
    };

    assert_outputs_match(&expected_array, &run_scan(), &mut ctx);

    // With layout reader caching enabled, a second scan re-uses the cached LayoutReader and must
    // produce the same result.
    if layout_reader_cache {
        assert_outputs_match(&expected_array, &run_scan(), &mut ctx);
    }

    Corpus::Keep
});

fn assert_outputs_match(
    expected_array: &ArrayRef,
    output_array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) {
    assert_eq!(
        expected_array.len(),
        output_array.len(),
        "Length was not preserved expected {} actual {}.",
        expected_array.len(),
        output_array.len()
    );
    assert_eq!(
        expected_array.dtype(),
        output_array.dtype(),
        "DTypes aren't preserved expected {}, actual {}.",
        expected_array.dtype(),
        output_array.dtype()
    );

    let bool_result = expected_array
        .binary(output_array.clone(), Operator::Eq)
        .vortex_expect("compare operation should succeed in fuzz test")
        .execute::<BoolArray>(ctx)
        .vortex_expect("execute bool");
    let true_count = bool_result.to_bit_buffer().true_count();
    if true_count != expected_array.len()
        && (bool_result
            .into_array()
            .all_valid(ctx)
            .vortex_expect("all_valid")
            || expected_array.all_valid(ctx).vortex_expect("all_valid"))
    {
        vortex_panic!(
            "Failed to match original array {}with{}",
            expected_array.display_tree(),
            output_array.display_tree()
        );
    }
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
