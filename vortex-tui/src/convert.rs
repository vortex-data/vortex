use std::path::Path;

use futures_util::StreamExt;
use indicatif::ProgressBar;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File;
use vortex::TryIntoArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect, VortexResult};
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamAdapter;

#[derive(Default)]
pub struct Flags {
    pub quiet: bool,
}

const BATCH_SIZE: usize = 8192;

/// Convert Parquet files to Vortex.
pub async fn exec_convert(input_path: impl AsRef<Path>, flags: Flags) -> VortexResult<()> {
    if !flags.quiet {
        eprintln!(
            "Converting input Parquet file: {}",
            input_path.as_ref().display()
        );
    }

    let output_path = input_path.as_ref().with_extension("vortex");
    let file = File::open(input_path).await?;

    let parquet = ParquetRecordBatchStreamBuilder::new(file)
        .await?
        .with_batch_size(BATCH_SIZE);
    let num_rows = parquet.metadata().file_metadata().num_rows();

    let dtype = DType::from_arrow(parquet.schema().as_ref());
    let mut vortex_stream = parquet
        .build()?
        .map(|record_batch| {
            record_batch
                .map_err(VortexError::from)
                .and_then(|rb| rb.try_into_array())
        })
        .boxed();

    if !flags.quiet {
        // Parquet reader returns batches, rather than row groups. So make sure we correctly
        // configure the progress bar.
        let nbatches = u64::try_from(num_rows)
            .vortex_expect("negative row count?")
            .div_ceil(BATCH_SIZE as u64);
        vortex_stream = ProgressBar::new(nbatches)
            .wrap_stream(vortex_stream)
            .boxed();
    }

    VortexWriteOptions::default()
        .write(
            File::create(output_path).await?,
            ArrayStreamAdapter::new(dtype, vortex_stream),
        )
        .await?;

    Ok(())
}
