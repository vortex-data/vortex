// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::execution::driver::ScanDriver;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_error::VortexExpect;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_io::runtime::BlockingRuntime;

use crate::FuzzFileAction;
use crate::RUNTIME;
use crate::SESSION;
use crate::file::has_duplicate_field_names;
use crate::file::has_nullable_struct;

/// Which scan implementation reads the written file back.
#[derive(Clone, Copy, Debug)]
pub enum FileScan {
    /// The layout-reader scan, `VortexFile::scan`.
    V1,
    /// The planner-driven scan, `ScanBuilder` run by `ScanDriver`.
    V2,
}

impl FileScan {
    fn read(
        self,
        file: &VortexFile,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
    ) -> Vec<ArrayRef> {
        match self {
            FileScan::V1 => file
                .scan()
                .vortex_expect("scan should succeed in fuzz test")
                .with_projection(projection)
                .with_some_filter(filter)
                .into_array_iter(&*RUNTIME)
                .vortex_expect("into_array_iter should succeed in fuzz test")
                .try_collect()
                .vortex_expect("collect should succeed in fuzz test"),
            FileScan::V2 => {
                let builder = ScanBuilder::new(SESSION.clone(), file.scan_file())
                    .with_projection(projection)
                    .with_some_filter(filter);
                RUNTIME
                    .block_on_stream(ScanDriver::new(RUNTIME.handle(), 2, true).run(builder))
                    .try_collect()
                    .vortex_expect("v2 scan should succeed in fuzz test")
            }
        }
    }
}

/// Writes the fuzzed array to a file, reads it back with `scan`, and checks the result against
/// filtering and projecting the array in memory. Returns `false` for inputs the file format does
/// not support.
pub fn run_file_io(fuzz: FuzzFileAction, scan: FileScan) -> bool {
    let FuzzFileAction {
        array,
        projection_expr,
        filter_expr,
        write_settings,
    } = fuzz;
    let array_data = array;

    if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype()) {
        return false;
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

    let write_options = write_settings.write_options();

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
    let mut output = scan.read(&file, projection, filter);

    let output_array = match output.len() {
        0 => Canonical::empty(expected_array.dtype()).into_array(),
        1 => output.pop().vortex_expect("one output"),
        _ => ChunkedArray::from_iter(output).into_array(),
    };

    assert_arrays_eq!(output_array, expected_array, &mut ctx);

    true
}
