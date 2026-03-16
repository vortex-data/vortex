// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::DatasetFixture;

/// 5×1000 rows sampled from deterministic random offsets in ClickBench hits partition 0.
/// Offsets (seed=42): [26225, 116739, 288389, 670487, 777572].
const CLICKBENCH_PARQUET: &[u8] = include_bytes!("../../../../data/clickbench_hits_5k.parquet");

struct ClickBenchHits5kFixture;

impl DatasetFixture for ClickBenchHits5kFixture {
    fn name(&self) -> &str {
        "clickbench_hits_5k"
    }

    fn description(&self) -> &str {
        "5000 rows (5x1000 from random offsets) of ClickBench hits dataset with wide schema of primitives and strings"
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let bytes = Bytes::from_static(CLICKBENCH_PARQUET);

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(|e| vortex_err!("failed to open parquet: {e}"))?
            .with_batch_size(1000)
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

pub fn fixtures() -> Vec<Box<dyn DatasetFixture>> {
    vec![Box::new(ClickBenchHits5kFixture)]
}
