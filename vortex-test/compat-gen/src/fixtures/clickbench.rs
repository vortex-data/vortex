// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::layout::LayoutId;
use vortex_array::ArrayRef;
use vortex_array::arrow::FromArrowArray;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::ExpectedEncoding;
use super::Fixture;

/// First partition of ClickBench hits, limited to 1000 rows.
const CLICKBENCH_URL: &str =
    "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_0.parquet";

const PARQUET_FILENAME: &str = "clickbench_hits_0.parquet";

pub struct ClickBenchHits1kFixture;

impl Fixture for ClickBenchHits1kFixture {
    fn name(&self) -> &str {
        "clickbench_hits_1k.vortex"
    }

    fn description(&self) -> &str {
        "First 1000 rows of ClickBench hits_0 partition (wide real-world schema)"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![
            ExpectedEncoding::Array(ArrayId::new_ref("vortex.primitive")),
            ExpectedEncoding::Array(ArrayId::new_ref("vortex.varbin")),
            ExpectedEncoding::Array(ArrayId::new_ref("vortex.struct")),
            ExpectedEncoding::Layout(LayoutId::new_ref("vortex.flat")),
            ExpectedEncoding::Layout(LayoutId::new_ref("vortex.struct")),
        ]
    }

    fn setup(&self, tmp_dir: &Path) -> VortexResult<()> {
        let parquet_path = tmp_dir.join(PARQUET_FILENAME);
        if parquet_path.exists() {
            return Ok(());
        }
        eprintln!("    downloading ClickBench parquet...");
        let bytes = reqwest::blocking::get(CLICKBENCH_URL)
            .map_err(|e| vortex_err!("failed to download ClickBench parquet: {e}"))?
            .bytes()
            .map_err(|e| vortex_err!("failed to read ClickBench response body: {e}"))?;
        std::fs::write(&parquet_path, &bytes)
            .map_err(|e| vortex_err!("failed to write parquet to tmp_dir: {e}"))?;
        Ok(())
    }

    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let parquet_path = tmp_dir.join(PARQUET_FILENAME);
        let file_bytes = std::fs::read(&parquet_path)
            .map_err(|e| vortex_err!("failed to read cached parquet: {e}"))?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(file_bytes))
            .map_err(|e| vortex_err!("failed to open parquet: {e}"))?
            .with_batch_size(1000)
            .with_limit(1000)
            .build()
            .map_err(|e| vortex_err!("failed to build parquet reader: {e}"))?;

        let batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| vortex_err!("failed to read parquet batches: {e}"))?;

        batches
            .into_iter()
            .map(|batch| ArrayRef::from_arrow(batch, false))
            .collect()
    }
}
