// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use anyhow::Result;
use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use tokio::runtime::Runtime;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect};
use vortex::file::VortexWriteOptions as WriteOptions;
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::stream::ArrayStream;

/// The tokio runtime for the write-side.
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::from)
        .vortex_expect("Failed to create tokio runtime")
});

pub(crate) struct VortexWriteOptions {
    inner: WriteOptions,
}

pub(crate) fn write_options_new() -> Box<VortexWriteOptions> {
    Box::new(VortexWriteOptions {
        inner: WriteOptions::default(),
    })
}

/// Convert an ArrowArrayStreamReader to a Vortex ArrayStream
fn arrow_stream_to_vortex_stream(reader: ArrowArrayStreamReader) -> Result<impl ArrayStream> {
    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|result| {
            result
                .map(|record_batch| ArrayRef::from_arrow(record_batch, false))
                .map_err(VortexError::from)
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
        let file = tokio::fs::File::create(path).await?;

        options.inner.write(file, vortex_stream).await?;
        Ok(())
    })
}
