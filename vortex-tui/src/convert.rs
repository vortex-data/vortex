#![allow(clippy::use_debug)]

use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use arrow_array::StructArray as ArrowStructArray;
use futures_util::Stream;
use indicatif::ProgressBar;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use pin_project::pin_project;
use tokio::fs::File;
use vortex::arrays::ChunkedArray;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexWriteOptions;
use vortex::stream::{ArrayStream, ArrayStreamArrayExt};
use vortex::{Array, ArrayRef};

#[derive(Default)]
pub struct Flags {
    pub quiet: bool,
}

/// Convert Parquet files to Vortex.
pub async fn exec_convert(input_path: impl AsRef<Path>, flags: Flags) -> VortexResult<()> {
    if !flags.quiet {
        eprintln!(
            "Converting input Parquet file: {}",
            input_path.as_ref().display()
        );
    }

    let wall_start = Instant::now();

    let output_path = input_path.as_ref().with_extension("vortex");
    let mut file = File::open(input_path).await?;

    let metadata =
        ArrowReaderMetadata::load_async(&mut file, ArrowReaderOptions::default()).await?;
    let has_root_level_nulls = metadata.parquet_schema().root_schema().is_optional();

    let mut reader = ParquetRecordBatchStreamBuilder::new(file).await?.build()?;
    let mut chunks = Vec::new();

    while let Some(mut reader) = reader.next_row_group().await? {
        for batch in reader.by_ref() {
            let batch = ArrowStructArray::from(batch?);
            let next_chunk = ArrayRef::from_arrow(&batch, has_root_level_nulls);
            chunks.push(next_chunk);
        }
    }

    let read_complete = Instant::now();

    if !flags.quiet {
        eprintln!(
            "Read Parquet file in {:?}",
            read_complete.duration_since(wall_start)
        );

        eprintln!(
            "Writing {} chunks to new Vortex file {}",
            chunks.len(),
            output_path.display()
        );
    }

    let dtype = chunks.first().vortex_expect("empty chunks").dtype().clone();
    let chunked_array = ChunkedArray::try_new(chunks, dtype)?;

    let writer = VortexWriteOptions::default();

    let output_file = File::create(output_path).await?;

    if !flags.quiet {
        let pb = ProgressBar::new(chunked_array.nchunks() as u64);
        let stream = ProgressArrayStream {
            inner: chunked_array.to_array_stream(),
            progress: pb,
        };
        writer.write(output_file, stream).await?;
    } else {
        writer
            .write(output_file, chunked_array.to_array_stream())
            .await?;
    }

    if !flags.quiet {
        eprintln!(
            "Wrote Vortex in {:?}",
            Instant::now().duration_since(read_complete)
        );
    }

    Ok(())
}

#[pin_project]
struct ProgressArrayStream<S> {
    #[pin]
    inner: S,
    progress: ProgressBar,
}

impl<S> Stream for ProgressArrayStream<S>
where
    S: Stream<Item = VortexResult<ArrayRef>>,
{
    type Item = VortexResult<ArrayRef>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match this.inner.poll_next(cx) {
            Poll::Ready(inner) => {
                this.progress.inc(1);
                Poll::Ready(inner)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> ArrayStream for ProgressArrayStream<S>
where
    S: ArrayStream,
{
    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }
}
