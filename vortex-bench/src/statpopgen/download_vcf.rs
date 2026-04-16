// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use noodles_vcf::Record;
use parquet::arrow::AsyncArrowWriter;
use reqwest::Client;
use tokio::fs::File;
use tokio::io::BufReader;
use tokio_stream::StreamExt;
use tokio_util::io::StreamReader;
use tracing::info;

use super::StatPopGenBenchmark;
use crate::idempotent_async;
use crate::statpopgen::builder::GnomADBuilder;
use crate::statpopgen::schema::schema_from_vcf_header;

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

                // The file is BIG so we only want to download a part of it
                let byte_stream = response
                    .bytes_stream()
                    .map(|x| x.map_err(std::io::Error::other));
                let stream_reader = StreamReader::new(byte_stream);

                // Wrap in BGZF reader
                let buf_reader = BufReader::new(stream_reader);
                let mut bgzf_reader = noodles_bgzf::r#async::io::Reader::new(buf_reader);

                // Read and parse VCF header
                let mut vcf_reader = noodles_vcf::r#async::io::Reader::new(&mut bgzf_reader);

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
                let mut builder = GnomADBuilder::new(&header, Arc::clone(&schema));
                let file = File::create(parquet_output_path).await?;
                let mut writer = AsyncArrowWriter::try_new(file, Arc::clone(&schema), None)
                    .context("Failed to create parquet writer")?;
                for i in progress.wrap_iter(0..self.n_rows) {
                    if i % ROW_GROUP_SIZE_IN_VARIANTS == 0 {
                        let rb = builder.finish()?;
                        builder = GnomADBuilder::new(&header, Arc::clone(&schema));
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
}
