// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use itertools::Itertools;
use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex::expr::lit;
use vortex::expr::root;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_error::VortexExpect;
use vortex_expr::ArrayExprExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_fuzz::CompressorStrategy;
use vortex_fuzz::FuzzFileAction;
use vortex_fuzz::RUNTIME;
use vortex_fuzz::SESSION;
use vortex_fuzz::has_duplicate_field_names;
use vortex_fuzz::has_nullable_struct;

fuzz_target!(|fuzz: FuzzFileAction| -> Corpus {
    let FuzzFileAction {
        array,
        projection_expr,
        filter_expr,
        compressor_strategy,
    } = fuzz;
    let array_data = array;

    if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype()) {
        return Corpus::Reject;
    }

    let mut ctx = SESSION.create_execution_ctx();
    let expected_array = {
        let bool_mask = array_data
            .clone()
            .apply(&filter_expr.clone().unwrap_or_else(|| lit(true)))
            .vortex_expect("filter expression evaluation should succeed in fuzz test");
        let bool_mask_bool = bool_mask
            .execute::<BoolArray>(&mut ctx)
            .vortex_expect("execute bool");
        let mask = bool_mask_bool.to_mask_fill_null_false(&mut ctx);
        let filtered = array_data
            .filter(mask)
            .vortex_expect("filter operation should succeed in fuzz test");
        filtered
            .apply(&projection_expr.clone().unwrap_or_else(root))
            .vortex_expect("projection expression evaluation should succeed in fuzz test")
    };

    let write_options = match compressor_strategy {
        CompressorStrategy::Default => SESSION.write_options(),
        CompressorStrategy::Compact => SESSION.write_options().with_strategy(
            WriteStrategyBuilder::from_session(&SESSION)
                .with_btrblocks_builder(
                    BtrBlocksCompressorBuilder::from_session(&SESSION).with_compact(),
                )
                .build(),
        ),
    };

    let mut full_buff = Vec::new();
    let _footer = write_options
        .blocking(&*RUNTIME)
        .write(&mut full_buff, array_data.to_array_iterator())
        .vortex_expect("file write should succeed in fuzz test");

    let file = SESSION
        .open_options()
        .open_buffer(full_buff)
        .vortex_expect("open_buffer should succeed in fuzz test");
    let projection = projection_expr
        .unwrap_or_else(root)
        .bind(file.dtype())
        .and_then(|expr| expr.optimize_recursive())
        .vortex_expect("projection should bind in fuzz test");
    let filter = filter_expr
        .map(|filter| {
            filter
                .bind(file.dtype())
                .and_then(|expr| expr.optimize_recursive())
        })
        .transpose()
        .vortex_expect("filter should bind in fuzz test");
    let mut output = file
        .scan()
        .vortex_expect("scan should succeed in fuzz test")
        .with_projection(projection)
        .with_some_filter(filter)
        .into_array_iter(&*RUNTIME)
        .vortex_expect("into_array_iter should succeed in fuzz test")
        .try_collect::<_, Vec<_>, _>()
        .vortex_expect("collect should succeed in fuzz test");

    let output_array = match output.len() {
        0 => Canonical::empty(expected_array.dtype()).into_array(),
        1 => output.pop().vortex_expect("one output"),
        _ => ChunkedArray::from_iter(output).into_array(),
    };

    assert_arrays_eq!(output_array, expected_array, &mut ctx);

    Corpus::Keep
});
