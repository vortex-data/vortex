// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Epoch C adapter — for Vortex v0.58.0 through HEAD
//
// Write: session.write_options(), returns WriteSummary, takes &mut sink
// Read:  session.open_options().open_buffer(buf) (sync), into_array_stream() (async)

use std::path::Path;
use std::sync::Arc;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::MaskFuture;
use vortex::array::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFn;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::supports_min_max;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

fn runtime() -> VortexResult<Runtime> {
    Runtime::new().map_err(|e| vortex_err!("failed to create tokio runtime: {e}"))
}

/// Compute all statistics on every node in the array tree.
///
/// The flat layout writer does not compute stats itself — it only serializes stats already
/// cached on each array node. This function walks the entire tree and forces computation of
/// all stats so they are present in the serialized output.
pub fn compute_all_stats(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let aggregates = [
        AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
        AggregateFn::new(Max, NumericalAggregateOpts::default()).erased(),
        AggregateFn::new(Sum, NumericalAggregateOpts::default()).erased(),
        AggregateFn::new(NullCount, EmptyOptions).erased(),
        AggregateFn::new(NanCount, EmptyOptions).erased(),
        AggregateFn::new(IsSorted, IsSortedOptions { strict: false }).erased(),
        AggregateFn::new(IsSorted, IsSortedOptions { strict: true }).erased(),
        AggregateFn::new(IsConstant, EmptyOptions).erased(),
        AggregateFn::new(UncompressedSizeInBytes, EmptyOptions).erased(),
    ];
    for node in array.depth_first_traversal() {
        let mut storage_dtype = node.dtype();
        while let DType::Extension(ext) = storage_dtype {
            storage_dtype = ext.storage_dtype();
        }
        let supports_extrema = supports_min_max(storage_dtype);

        for aggregate in &aggregates {
            let is_extremum = aggregate.is::<Min>() || aggregate.is::<Max>();
            if aggregate.return_dtype(node.dtype()).is_some() && (!is_extremum || supports_extrema)
            {
                node.aggregations().compute_result(aggregate, ctx)?;
            }
        }
    }
    Ok(())
}

/// Write a sequence of array chunks as a `.vortex` file with no compression.
///
/// Uses `FlatLayoutStrategy` directly — no repartitioning, no zone maps, no dictionary
/// encoding, no compression. Each chunk is serialized as a single flat segment.
pub fn write_file(path: &Path, chunk: ArrayRef) -> VortexResult<()> {
    write_compressed(path, chunk, Arc::new(FlatLayoutStrategy::default()))
}

/// Write a sequence of array chunks to an in-memory `.vortex` byte buffer with no compression.
pub fn write_file_to_bytes(chunk: ArrayRef) -> VortexResult<ByteBuffer> {
    write_compressed_to_bytes(chunk, Arc::new(FlatLayoutStrategy::default()))
}

/// Write a `.vortex` file using a caller-provided layout strategy (compressor pipeline).
///
/// The write is restricted to the default session's enabled editions, so a fixture cannot record
/// an encoding that carries no read-compatibility guarantee.
pub fn write_compressed(
    path: &Path,
    chunk: ArrayRef,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<()> {
    let stream = ArrayStreamAdapter::new(chunk.dtype().clone(), stream::iter([Ok(chunk)]));

    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| vortex_err!("failed to create {}: {e}", path.display()))?;
        let _summary = session
            .write_options()
            .with_strategy(strategy)
            .write(&mut file, stream)
            .await?;
        Ok(())
    })
}

/// Write a `.vortex` file into an in-memory byte buffer using a caller-provided layout strategy.
pub fn write_compressed_to_bytes(
    chunk: ArrayRef,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<ByteBuffer> {
    write_compressed_to_bytes_with_session(&VortexSession::default(), chunk, strategy)
}

/// Write a `.vortex` file into memory using a caller-provided session and layout strategy.
///
/// The write is restricted to the session's enabled editions.
pub fn write_compressed_to_bytes_with_session(
    session: &VortexSession,
    chunk: ArrayRef,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<ByteBuffer> {
    let stream = ArrayStreamAdapter::new(chunk.dtype().clone(), stream::iter([Ok(chunk)]));
    let session = session.clone();

    runtime()?.block_on(async move {
        let session = session.with_tokio();
        let mut bytes = Vec::new();
        let _summary = session
            .write_options()
            .with_strategy(strategy)
            .write(&mut bytes, stream)
            .await?;
        Ok(ByteBuffer::from(bytes))
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
pub fn read_file(bytes: ByteBuffer) -> VortexResult<ArrayRef> {
    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let file = session.open_options().open_buffer(bytes)?;
        file.scan()?.into_array_stream()?.read_all().await
    })
}

/// Open a `.vortex` file and fully decode every array in the layout tree, including
/// auxiliary data like zone maps and dictionaries.
///
/// Walks the entire layout tree and for each leaf `FlatLayout`, reads the segment
/// and calls `ArrayParts::decode()` to fully deserialize the array. This exercises
/// every segment in the file — not just the data path that a plain `scan()` touches.
/// If any segment is corrupt or any array fails to decode, this will error.
pub fn read_layout_tree(bytes: ByteBuffer) -> VortexResult<()> {
    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let file = session.open_options().open_buffer(bytes)?;
        let root_layout = Arc::clone(file.footer().layout());
        let segment_source = file.segment_source();

        for layout_result in root_layout.depth_first_traversal() {
            let layout = layout_result?;
            if layout.as_opt::<Flat>().is_none() {
                continue;
            }
            let row_count = layout.row_count();
            if row_count == 0 {
                continue;
            }
            let reader = layout.new_reader(
                "".into(),
                Arc::clone(&segment_source),
                &session,
                &Default::default(),
            )?;
            let len =
                usize::try_from(row_count).map_err(|e| vortex_err!("row count overflow: {e}"))?;
            let expr = root().bind(reader.dtype())?;
            reader
                .projection_evaluation(&(0..row_count), &expr, MaskFuture::new_true(len))?
                .await?;
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::expr::stats::Precision;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn compute_all_stats_nested_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements = buffer![1i32, 2, 3, 4].into_array();
        let inner = FixedSizeListArray::try_new(elements.clone(), 2, Validity::NonNullable, 2)?
            .into_array();
        let outer = ListArray::try_new(
            inner.clone(),
            buffer![0u32, 1, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        compute_all_stats(&outer, &mut ctx)?;

        let min = AggregateFn::new(Min, NumericalAggregateOpts::default()).erased();
        let max = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();
        for node in [&outer, &inner] {
            assert_eq!(node.aggregations().get_result(&min), Precision::Absent);
            assert_eq!(node.aggregations().get_result(&max), Precision::Absent);
        }
        assert_eq!(
            elements.aggregations().get_result_as::<i32>(&min)?,
            Precision::Exact(1)
        );
        assert_eq!(
            elements.aggregations().get_result_as::<i32>(&max)?,
            Precision::Exact(4)
        );
        Ok(())
    }
}
