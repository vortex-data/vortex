// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrow::FromArrowArray;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::ArrayFixture;

/// First partition of ClickBench hits, limited to 1000 rows.
const CLICKBENCH_URL: &str =
    "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_0.parquet";

struct ClickBenchHits1kFixture;

impl ArrayFixture for ClickBenchHits1kFixture {
    fn name(&self) -> &str {
        "clickbench_hits_1k.vortex"
    }

    fn description(&self) -> &str {
        "First 1000 rows of ClickBench hits dataset with wide schema of primitives and strings"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Struct::ID, Primitive::ID, VarBin::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let bytes = reqwest::blocking::get(CLICKBENCH_URL)
            .map_err(|e| vortex_err!("failed to download ClickBench parquet: {e}"))?
            .bytes()
            .map_err(|e| vortex_err!("failed to read ClickBench response body: {e}"))?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(|e| vortex_err!("failed to open parquet: {e}"))?
            .with_batch_size(1000)
            .with_limit(1000)
            .build()
            .map_err(|e| vortex_err!("failed to build parquet reader: {e}"))?;

        let batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| vortex_err!("failed to read parquet batches: {e}"))?;

        Ok(ChunkedArray::from_iter(
            batches
                .into_iter()
                .map(|batch| ArrayRef::from_arrow(batch, false))
                .collect::<VortexResult<Vec<_>>>()?,
        )
        .into_array())
    }
}

pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    vec![Box::new(ClickBenchHits1kFixture)]
}
