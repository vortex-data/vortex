// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::array::stream::ArrayStream;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::file::BlockingWriter;
use vortex::file::VortexWriteOptions as WriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::expr::stats::Stat;
use vortex::io::VortexWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::layout::LayoutStrategy;

use crate::ffi;
use crate::RUNTIME;
use crate::SESSION;
use crate::session::VortexSession;

pub(crate) struct VortexWriteStrategy {
    inner: Arc<dyn LayoutStrategy>,
}

pub(crate) struct VortexWriteStrategyBuilder {
    inner: WriteStrategyBuilder,
}

pub(crate) struct VortexWriteOptions {
    inner: WriteOptions,
}

pub(crate) struct VortexWriter {
    options: Option<WriteOptions>,
    path: String,
    inner: Option<BlockingWriter<'static, 'static, CurrentThreadRuntime>>,
}

pub(crate) fn write_options_new() -> Box<VortexWriteOptions> {
    Box::new(VortexWriteOptions {
        inner: SESSION.write_options(),
    })
}

pub(crate) fn write_options_new_with_session(session: &VortexSession) -> Box<VortexWriteOptions> {
    Box::new(VortexWriteOptions {
        inner: WriteOptions::new(session.inner.clone()),
    })
}

pub(crate) fn write_strategy_builder_new() -> Box<VortexWriteStrategyBuilder> {
    Box::new(VortexWriteStrategyBuilder {
        inner: WriteStrategyBuilder::default(),
    })
}

pub(crate) fn write_options_exclude_dtype(options: &mut VortexWriteOptions) {
    take_mut::take(&mut options.inner, WriteOptions::exclude_dtype);
}

pub(crate) fn write_strategy_builder_with_row_block_size(
    builder: &mut VortexWriteStrategyBuilder,
    row_block_size: usize,
) -> Result<()> {
    take_mut::take(&mut builder.inner, |inner| inner.with_row_block_size(row_block_size));
    Ok(())
}

#[cfg(feature = "zstd")]
pub(crate) fn write_strategy_builder_with_compact_encodings(
    builder: &mut VortexWriteStrategyBuilder,
) -> Result<()> {
    take_mut::take(&mut builder.inner, WriteStrategyBuilder::with_compact_encodings);
    Ok(())
}

#[cfg(not(feature = "zstd"))]
pub(crate) fn write_strategy_builder_with_compact_encodings(
    _builder: &mut VortexWriteStrategyBuilder,
) -> Result<()> {
    anyhow::bail!("Compact encodings require building with zstd");
}

pub(crate) fn write_strategy_builder_build(
    builder: Box<VortexWriteStrategyBuilder>,
) -> Box<VortexWriteStrategy> {
    Box::new(VortexWriteStrategy {
        inner: builder.inner.build(),
    })
}

pub(crate) fn write_options_with_strategy(
    options: &mut VortexWriteOptions,
    strategy: &VortexWriteStrategy,
) {
    take_mut::take(&mut options.inner, |inner| {
        inner.with_strategy(strategy.inner.clone())
    });
}

fn file_stat_to_stat(stat: ffi::FileStat) -> Result<Stat> {
    Ok(match stat {
        ffi::FileStat::IsConstant => Stat::IsConstant,
        ffi::FileStat::IsSorted => Stat::IsSorted,
        ffi::FileStat::IsStrictSorted => Stat::IsStrictSorted,
        ffi::FileStat::Max => Stat::Max,
        ffi::FileStat::Min => Stat::Min,
        ffi::FileStat::Sum => Stat::Sum,
        ffi::FileStat::NullCount => Stat::NullCount,
        ffi::FileStat::UncompressedSizeInBytes => Stat::UncompressedSizeInBytes,
        ffi::FileStat::NaNCount => Stat::NaNCount,
        _ => anyhow::bail!("unknown file stat value"),
    })
}

pub(crate) fn write_options_with_file_statistics(
    options: &mut VortexWriteOptions,
    statistics: &[ffi::FileStat],
) -> Result<()> {
    let file_statistics = statistics
        .iter()
        .copied()
        .map(file_stat_to_stat)
        .collect::<Result<Vec<_>>>()?;
    take_mut::take(&mut options.inner, |inner| {
        inner.with_file_statistics(file_statistics)
    });
    Ok(())
}

pub(crate) fn write_options_without_file_statistics(options: &mut VortexWriteOptions) {
    take_mut::take(&mut options.inner, |inner| inner.with_file_statistics(vec![]));
}

pub(crate) fn write_options_into_writer(
    options: Box<VortexWriteOptions>,
    path: &str,
) -> Box<VortexWriter> {
    Box::new(VortexWriter {
        options: Some(options.inner),
        path: path.to_string(),
        inner: None,
    })
}

/// Convert an ArrowArrayStreamReader to a Vortex ArrayStream
fn arrow_stream_to_vortex_stream(reader: ArrowArrayStreamReader) -> Result<impl ArrayStream> {
    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|result| {
            result
                .map_err(VortexError::from)
                .and_then(|record_batch| ArrayRef::from_arrow(record_batch, false))
        }),
    );

    Ok(array_iter.into_array_stream())
}

/// # Safety
///
/// input_stream should be valid FFI_ArrowArrayStream.
/// See [`FFI_ArrowArrayStream::from_raw`]
pub(crate) unsafe fn write_array_stream(
    options: Box<VortexWriteOptions>,
    input_stream: *mut u8,
    path: &str,
) -> Result<()> {
    let path = path.to_string();

    let stream_reader =
        unsafe { ArrowArrayStreamReader::from_raw(input_stream as *mut FFI_ArrowArrayStream) }?;

    let vortex_stream = arrow_stream_to_vortex_stream(stream_reader)?;

    RUNTIME.block_on(async {
        let mut file = async_fs::File::create(path).await?;
        options.inner.write(&mut file, vortex_stream).await?;
        file.shutdown().await?;
        Ok(())
    })
}

impl VortexWriter {
    fn init_if_needed(&mut self, dtype: DType) -> Result<()> {
        if self.inner.is_some() {
            return Ok(());
        }

        let options = self
            .options
            .take()
            .ok_or_else(|| anyhow::anyhow!("writer options were already consumed"))?;
        let file = std::fs::File::create(&self.path)?;
        self.inner = Some(options.blocking(&*RUNTIME).writer(file, dtype));
        Ok(())
    }
}

/// # Safety
///
/// input_stream should be valid FFI_ArrowArrayStream.
/// See [`FFI_ArrowArrayStream::from_raw`]
pub(crate) unsafe fn writer_push_array_stream(
    writer: &mut VortexWriter,
    input_stream: *mut u8,
) -> Result<()> {
    let stream_reader =
        unsafe { ArrowArrayStreamReader::from_raw(input_stream as *mut FFI_ArrowArrayStream) }?;

    if writer.inner.is_none() {
        writer.init_if_needed(DType::from_arrow(stream_reader.schema()))?;
    }

    let inner = writer
        .inner
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("writer was not initialized"))?;

    for batch in stream_reader {
        let record_batch = batch?;
        let chunk = ArrayRef::from_arrow(record_batch, false)?;
        inner.push(chunk)?;
    }

    Ok(())
}

pub(crate) fn writer_bytes_written(writer: &VortexWriter) -> u64 {
    writer.inner.as_ref().map_or(0, |w| w.bytes_written())
}

pub(crate) fn writer_buffered_bytes(writer: &VortexWriter) -> u64 {
    writer.inner.as_ref().map_or(0, |w| w.buffered_bytes())
}

pub(crate) fn writer_finish(mut writer: Box<VortexWriter>) -> Result<()> {
    let inner = writer
        .inner
        .take()
        .ok_or_else(|| anyhow::anyhow!("cannot finish writer before first push"))?;
    inner.finish()?;
    Ok(())
}
