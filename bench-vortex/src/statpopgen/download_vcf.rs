// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use noodles_vcf::Record;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use reqwest::Client;
use tokio::fs::{File, create_dir_all};
use tokio::io::BufReader;
use tokio_util::io::StreamReader;
use tracing::info;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::file::{VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamAdapter;

use super::StatPopGenBenchmark;
use crate::statpopgen::builder::GnomADBuilder;
use crate::statpopgen::schema::schema_from_vcf_header;
use crate::{Format, idempotent_async};

// DuckDB parallelizes parquet at row-group granularity. Each of our rows are quite big (~4000
// genotypes each with tens of bytes of data).
const ROW_GROUP_SIZE_IN_VARIANTS: u64 = 1024;

impl StatPopGenBenchmark {
    pub async fn download_parquet(&self) -> Result<()> {
        let url = format!(
            "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/genomes/{}.vcf.bgz",
            StatPopGenBenchmark::FILE_NAME
        );
        let parquet_output_path = self.parquet_path()?;
        idempotent_async(
            &parquet_output_path,
            async |parquet_output_path| -> Result<()> {
                info!(
                    "Downloading first {} lines of gnomAD v3.1.2 HGDP-1kG chr21.",
                    self.n_rows
                );

                // Fetch the remote stream
                let client = Client::new();
                let response = client
                    .get(url)
                    .send()
                    .await
                    .context("reqwest failed")?
                    .error_for_status()
                    .context("reqwest bad status")?;
                let byte_stream = response.bytes_stream().map(|x| x.map_err(io::Error::other));
                let stream_reader = StreamReader::new(byte_stream);

                // Wrap in BGZF reader
                let buf_reader = BufReader::new(stream_reader);
                let mut bgzf_reader = noodles_bgzf::r#async::io::Reader::new(buf_reader);

                // Read and parse VCF header
                let mut vcf_reader = noodles_vcf::AsyncReader::new(&mut bgzf_reader);

                // Read and print the first 100,000 records
                let header = vcf_reader.read_header().await?;
                let progress = ProgressBar::new(self.n_rows);
                progress.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}",
                    )
                    .expect("style is ok"),
                );
                let mut record = Record::default();
                let schema = schema_from_vcf_header(&header);
                let mut builder = GnomADBuilder::new(&header, schema.clone());
                let file = File::create(parquet_output_path).await?;
                let mut writer = AsyncArrowWriter::try_new(file, schema.clone(), None)
                    .context("Failed to create parquet writer")?;
                for i in progress.wrap_iter(0..self.n_rows) {
                    if i % ROW_GROUP_SIZE_IN_VARIANTS == 0 {
                        let rb = builder.finish()?;
                        builder = GnomADBuilder::new(&header, schema.clone());
                        writer
                            .write(&rb)
                            .await
                            .context("Failed to create parquet writer")?;
                        writer
                            .flush()
                            .await
                            .context("Failed to create parquet writer")?;
                    }

                    let bytes_read = vcf_reader.read_record(&mut record).await?;
                    if bytes_read == 0 {
                        bail!("Reached end of stream after only {} records.", i)
                    }
                    builder.consume_record(&header, &mut record)?;
                }

                let rb = builder.finish()?;
                writer
                    .write(&rb)
                    .await
                    .context("Failed to create parquet writer")?;
                writer
                    .flush()
                    .await
                    .context("Failed to create parquet writer")?;

                writer
                    .close()
                    .await
                    .context("Failed to create parquet writer")?;

                info!(
                    "Finished downloading first {} lines of gnomAD v3.1.2 HGDP-1kG chr21.",
                    self.n_rows
                );

                Ok(())
            },
        )
        .await?;
        Ok(())
    }

    pub async fn parquet_to_vortex(&self, format: Format) -> Result<()> {
        let parquet_path = self.parquet_path()?;
        let strategy = WriteStrategyBuilder::new();
        let (output_path, strategy) = match format {
            Format::OnDiskVortex => (self.vortex_path()?, strategy),
            Format::VortexCompact => (
                self.vortex_compact_path()?,
                strategy.with_compressor(CompactCompressor::default()),
            ),
            otherwise => {
                bail!("you asked for vortex but gave me {}", otherwise)
            }
        };

        idempotent_async(&output_path, async |output_path| -> Result<_> {
            info!("Converting StatPopGen dataset from Parquet to {}.", format);

            create_dir_all(
                &output_path
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("vortex path must be a file in a directory"))?,
            )
            .await?;
            let file = File::open(parquet_path).await?;

            let parquet = ParquetRecordBatchStreamBuilder::new(file)
                .await
                .context("Failed to create parquet writer")?;
            let num_groups = parquet.metadata().num_row_groups();

            let dtype = DType::from_arrow(parquet.schema().as_ref());
            let mut vortex_stream = parquet
                .build()
                .map_err(|err| VortexError::generic(Box::new(err)))?
                .map(|record_batch| {
                    record_batch
                        .map_err(|err| VortexError::generic(Box::new(err)))
                        .map(|rb| ArrayRef::from_arrow(rb, false))
                })
                .boxed();

            // Parquet reader returns batches, rather than row groups. So make sure we correctly
            // configure the progress bar.
            vortex_stream = ProgressBar::new(num_groups as u64)
                .wrap_stream(vortex_stream)
                .boxed();

            VortexWriteOptions::default()
                .with_strategy(strategy.build())
                .write(
                    &mut File::create(output_path).await?,
                    ArrayStreamAdapter::new(dtype, vortex_stream),
                )
                .await?;

            info!(
                "Finished converting StatPopGen dataset from Parquet to {}.",
                format
            );

            Ok(())
        })
        .await?;

        Ok(())
    }
}
