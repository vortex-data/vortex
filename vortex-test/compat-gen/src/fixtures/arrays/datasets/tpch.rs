// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_arrow::ArrowSession;
use vortex_error::VortexResult;

use crate::fixtures::DatasetFixture;

const SCALE_FACTOR: f64 = 0.01;

fn collect_batches_as_vortex(
    iter: impl RecordBatchReader,
    arrow: &ArrowSession,
) -> VortexResult<ArrayRef> {
    let batches = iter.collect::<Result<Vec<RecordBatch>, _>>()?;
    Ok(ChunkedArray::from_iter(
        batches
            .into_iter()
            .map(|batch| {
                let schema = batch.schema();
                arrow.from_arrow_record_batch(batch, &schema)
            })
            .collect::<VortexResult<Vec<_>>>()?,
    )
    .into_array())
}

struct TpchLineitemFixture;

impl DatasetFixture for TpchLineitemFixture {
    fn name(&self) -> &str {
        "tpch_lineitem"
    }

    fn description(&self) -> &str {
        "TPC-H lineitem table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn build(&self, arrow: &ArrowSession) -> VortexResult<ArrayRef> {
        let generator = LineItemGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::LineItemArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter, arrow)
    }
}

struct TpchOrdersFixture;

impl DatasetFixture for TpchOrdersFixture {
    fn name(&self) -> &str {
        "tpch_orders"
    }

    fn description(&self) -> &str {
        "TPC-H orders table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn build(&self, arrow: &ArrowSession) -> VortexResult<ArrayRef> {
        let generator = OrderGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::OrderArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter, arrow)
    }
}

pub fn fixtures() -> Vec<Box<dyn DatasetFixture>> {
    vec![Box::new(TpchLineitemFixture), Box::new(TpchOrdersFixture)]
}
