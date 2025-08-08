// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::StreamExt;
use indicatif::ProgressBar;
use noodles_vcf::Record;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use reqwest::Client;
use std::io;
use std::sync::Arc;
use tokio::fs::{File, create_dir_all};
use tokio::io::BufReader;
use tokio::runtime::Handle;
use tokio_util::io::StreamReader;
use tracing::info;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect, VortexResult};
use vortex::error::{vortex_bail, vortex_err};
use vortex::file::VortexWriteOptions;
use vortex::file::WriteStrategyBuilder;
use vortex::stream::ArrayStreamAdapter;

use crate::Format;
use crate::idempotent_async;
use crate::statpopgen::builder::GnomADBuilder;
use crate::statpopgen::schema::SCHEMA;

use super::StatPopGenBenchmark;

const ROW_GROUP_SIZE_IN_VARIANTS: u64 = 1 << 14;

impl StatPopGenBenchmark {
    pub async fn download_parquet(&self) -> VortexResult<()> {
        let url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vcf.bgz";
        let parquet_output_path = self.parquet_path()?;
        idempotent_async(&parquet_output_path, async |parquet_output_path| {
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
                .map_err(|err| vortex_err!("reqwest failed: {err}"))?
                .error_for_status()
                .map_err(|err| vortex_err!("reqwest bad status: {err}"))?;
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
            let mut record = Record::default();
            let mut builder = GnomADBuilder::new();
            let file = File::create(parquet_output_path).await?;
            let mut writer = AsyncArrowWriter::try_new(file, SCHEMA.clone(), None)?;
            for i in progress.wrap_iter(0..self.n_rows) {
                if i == ROW_GROUP_SIZE_IN_VARIANTS {
                    let rb = builder.finish()?;
                    builder = GnomADBuilder::new();
                    writer.write(&rb).await?;
                    writer.flush().await?;
                }

                let bytes_read = vcf_reader.read_record(&mut record).await?;
                if bytes_read == 0 {
                    vortex_bail!("Reached end of stream after only {} records.", i)
                }
                builder.consume_record(&header, &mut record)?;
            }

            let rb = builder.finish()?;
            writer.write(&rb).await?;
            writer.flush().await?;

            writer.close().await?;

            Ok(())
        })
        .await?;
        Ok(())
    }

    const BATCH_SIZE: usize = 8192;

    pub async fn parquet_to_vortex(&self, format: Format) -> VortexResult<()> {
        let parquet_path = self.parquet_path()?;
        let strategy = WriteStrategyBuilder::new().with_executor(Arc::new(Handle::current()));
        let (output_path, strategy) = match format {
            Format::OnDiskVortex => {
                info!("Converting StatPopGen dataset from Parquet to Vortex.");
                (self.vortex_path()?, strategy)
            }
            Format::VortexCompact => {
                info!("Converting StatPopGen dataset from Parquet to Vortex-compact.");
                (
                    self.vortex_compact_path()?,
                    strategy.with_compressor(CompactCompressor::default()),
                )
            }
            otherwise => {
                vortex_bail!("you asked for vortex but gave me {}", otherwise)
            }
        };

        create_dir_all(
            &output_path
                .parent()
                .ok_or_else(|| vortex_err!("vortex path must be a file in a directory"))?,
        )
        .await?;
        let file = File::open(parquet_path).await?;

        let parquet = ParquetRecordBatchStreamBuilder::new(file)
            .await?
            .with_batch_size(Self::BATCH_SIZE);
        let num_rows = parquet.metadata().file_metadata().num_rows();

        let dtype = DType::from_arrow(parquet.schema().as_ref());
        let mut vortex_stream = parquet
            .build()?
            .map(|record_batch| {
                record_batch
                    .map_err(VortexError::from)
                    .map(|rb| ArrayRef::from_arrow(rb, false))
            })
            .boxed();

        // Parquet reader returns batches, rather than row groups. So make sure we correctly
        // configure the progress bar.
        let nbatches = u64::try_from(num_rows)
            .vortex_expect("negative row count?")
            .div_ceil(Self::BATCH_SIZE as u64);
        vortex_stream = ProgressBar::new(nbatches)
            .wrap_stream(vortex_stream)
            .boxed();

        VortexWriteOptions::default()
            .with_strategy(strategy.build())
            .write(
                File::create(output_path).await?,
                ArrayStreamAdapter::new(dtype, vortex_stream),
            )
            .await?;

        Ok(())
    }

    pub fn parquet_to_duckdb(&self) -> VortexResult<()> {
        todo!()
    }
}
