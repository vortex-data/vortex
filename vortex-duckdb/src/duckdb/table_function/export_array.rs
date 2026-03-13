use std::sync::Arc;

use vortex::array::Canonical;
use vortex::array::DynArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ScalarFnVTable;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::error::VortexExpect;
use vortex::scalar_fn::fns::pack::Pack;

use crate::SESSION;
use crate::cpp::duckdb_data_chunk;
use crate::duckdb::DataChunk;
use crate::duckdb::TableFunction;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

// TODO In the original implementation, exporter is initialized in the
// local state, and conversion cache is scoped per partition.

pub(crate) unsafe extern "C-unwind" fn export_array_callback<T: TableFunction>(
    array: *const crate::cpp::vx_array,
    chunk: duckdb_data_chunk,
) -> u64 {
    let chunk = unsafe { DataChunk::borrow_mut(chunk) };
    let mut batch_id = u64::MAX;
    if array.is_null() {
        return batch_id;
    }
    let array_result: Arc<dyn DynArray> =
        vortex_ffi::vx_array::as_ref(array as *const vortex_ffi::vx_array).clone();

    // TODO this will produce incorrect results as exporter may export
    // multiple data chunks. This exporter exports only one data chunk

    let conversion_cache = ConversionCache::default();

    let mut ctx = SESSION.create_execution_ctx();
    let array_result = array_result
        .optimize_recursive()
        .vortex_expect("failed to optimize array");

    let array_result = if let Some(array) = array_result.as_opt::<Struct>() {
        array.clone()
    } else if let Some(array) = array_result.as_opt::<ScalarFnVTable>()
        && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        StructArray::new(
            pack_options.names.clone(),
            array.children(),
            array.len(),
            pack_options.nullability.into(),
        )
    } else {
        array_result
            .execute::<Canonical>(&mut ctx)
            .vortex_expect("failed to canonicalize array")
            .into_struct()
    };

    let mut exporter = ArrayExporter::try_new(&array_result, &conversion_cache, ctx)
        .vortex_expect("failed to initialize array exporter");

    // Relaxed since there is no intra-instruction ordering required.
    //batch_id = Some(global_state.batch_id.fetch_add(1, Ordering::Relaxed));
    batch_id += 1;

    let has_more_data = exporter
        .export(chunk)
        .vortex_expect("failed to export chunk");
    //global_state
    //    .bytes_read
    //    .fetch_add(chunk.len(), Ordering::Relaxed);

    if !has_more_data {
        // This exporter is fully consumed.
        //EXPORTER = None;
        batch_id = u64::MAX;
    }

    assert!(!chunk.is_empty());
    batch_id
}
