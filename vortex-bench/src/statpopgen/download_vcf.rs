// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::pin::pin;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use noodles_bgzf::r#async::io::Reader as BgzfAsyncReader;
use noodles_vcf::Header;
use noodles_vcf::Record;
use noodles_vcf::r#async::io::Reader as VcfAsyncReader;
use noodles_vcf::io::Reader as VcfReader;
use parquet::arrow::AsyncArrowWriter;
use reqwest::Client;
use reqwest::StatusCode;
use reqwest::header::CONTENT_LENGTH;
use reqwest::header::RANGE;
use tokio::fs::File;
use tokio::io::AsyncBufRead;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;
use tracing::info;
use vortex::utils::parallelism::get_available_parallelism;

use super::StatPopGenBenchmark;
use crate::idempotent_async;
use crate::statpopgen::builder::GnomADBuilder;
use crate::statpopgen::schema::schema_from_vcf_header;

// DuckDB parallelizes parquet at row-group granularity. Each of our rows are quite big (~4000
// genotypes each with tens of bytes of data).
const ROW_GROUP_SIZE_IN_VARIANTS: u64 = 1024;

// A single S3 GET stream tops out far below the runner's bandwidth, so the compressed file is
// fetched as concurrent byte ranges that are reassembled in order.
const RANGE_REQUEST_BYTES: u64 = 16 << 20;
const CONCURRENT_RANGE_REQUESTS: usize = 16;

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

                let client = Client::new();
                let byte_stream = ranged_byte_stream(client, url).await?;

                // BGZF blocks are inflated concurrently by the async reader.
                let bgzf_reader = BgzfAsyncReader::new(StreamReader::new(byte_stream));
                let mut vcf_reader = VcfAsyncReader::new(bgzf_reader);
                let header = Arc::new(vcf_reader.read_header().await?);
                let bgzf_reader = vcf_reader.into_inner();

                let progress = ProgressBar::new(self.n_rows);
                progress.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {per_sec}",
                    )
                    .expect("style is ok"),
                );
                let schema = schema_from_vcf_header(&header);
                let file = File::create(parquet_output_path).await?;
                let mut writer = AsyncArrowWriter::try_new(file, Arc::clone(&schema), None)
                    .context("Failed to create parquet writer")?;

                // Splitting lines is cheap and stays serial; parsing each row group's lines into
                // a record batch is the expensive part and runs on the blocking pool.
                let n_rows = self.n_rows;
                let chunks = stream::try_unfold(
                    (bgzf_reader, 0u64),
                    move |(mut reader, rows_read)| async move {
                        if rows_read == n_rows {
                            return Ok(None);
                        }
                        let n_lines = ROW_GROUP_SIZE_IN_VARIANTS.min(n_rows - rows_read);
                        let lines = read_lines(&mut reader, n_lines)
                            .await
                            .with_context(|| format!("Failed reading records after {rows_read}"))?;
                        Ok::<_, anyhow::Error>(Some((
                            (lines, n_lines),
                            (reader, rows_read + n_lines),
                        )))
                    },
                );
                let mut batches = pin!(
                    chunks
                        .map_ok(|(lines, n_lines)| {
                            let header = Arc::clone(&header);
                            let schema = Arc::clone(&schema);
                            async move {
                                let rb = tokio::task::spawn_blocking(move || {
                                    parse_lines(&header, schema, &lines)
                                })
                                .await??;
                                Ok::<_, anyhow::Error>((rb, n_lines))
                            }
                        })
                        .try_buffered(get_available_parallelism().unwrap_or(1))
                );

                while let Some((rb, n_lines)) = batches.try_next().await? {
                    // Flushing after every batch makes each one its own row group.
                    writer
                        .write(&rb)
                        .await
                        .context("Failed to write parquet batch")?;
                    writer
                        .flush()
                        .await
                        .context("Failed to flush parquet row group")?;
                    progress.inc(n_lines);
                }
                progress.finish();

                writer
                    .close()
                    .await
                    .context("Failed to close parquet writer")?;

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

/// Stream the object at `url` as in-order chunks fetched by concurrent range requests.
///
/// Requests are only issued as the stream is polled, so a consumer that stops early leaves at most
/// [`CONCURRENT_RANGE_REQUESTS`] ranges in flight.
async fn ranged_byte_stream(
    client: Client,
    url: String,
) -> Result<impl Stream<Item = io::Result<Bytes>>> {
    let head = client
        .head(&url)
        .send()
        .await
        .context("reqwest HEAD failed")?
        .error_for_status()
        .context("reqwest HEAD bad status")?;
    // `Response::content_length` reports the (empty) body of a HEAD response, so read the header.
    let len: u64 = head
        .headers()
        .get(CONTENT_LENGTH)
        .context("HEAD response has no Content-Length")?
        .to_str()?
        .parse()
        .context("invalid Content-Length")?;

    Ok(stream::iter(0..len.div_ceil(RANGE_REQUEST_BYTES))
        .map(move |idx| {
            let start = idx * RANGE_REQUEST_BYTES;
            let client = client.clone();
            let url = url.clone();
            async move {
                let end = (start + RANGE_REQUEST_BYTES).min(len) - 1;
                let response = client
                    .get(url)
                    .header(RANGE, format!("bytes={start}-{end}"))
                    .send()
                    .await
                    .and_then(|r| r.error_for_status())
                    .map_err(io::Error::other)?;
                if response.status() != StatusCode::PARTIAL_CONTENT {
                    return Err(io::Error::other(format!(
                        "expected 206 for range {start}-{end}, got {}",
                        response.status()
                    )));
                }
                response.bytes().await.map_err(io::Error::other)
            }
        })
        .buffered(CONCURRENT_RANGE_REQUESTS))
}

/// Read exactly `n_lines` newline-terminated lines from `reader`.
async fn read_lines<R: AsyncBufRead + Unpin>(reader: &mut R, n_lines: u64) -> Result<Vec<u8>> {
    let mut lines = Vec::new();
    for i in 0..n_lines {
        if reader.read_until(b'\n', &mut lines).await? == 0 {
            bail!("Reached end of stream after {i} more records.");
        }
    }
    Ok(lines)
}

fn parse_lines(header: &Header, schema: SchemaRef, lines: &[u8]) -> Result<RecordBatch> {
    let mut reader = VcfReader::new(lines);
    let mut record = Record::default();
    let mut builder = GnomADBuilder::new(header, schema);
    while reader.read_record(&mut record)? != 0 {
        builder.consume_record(header, &mut record)?;
    }
    Ok(builder.finish()?)
}
