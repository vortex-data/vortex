// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen_arrow::RecordBatchIterator;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Extension;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBinView;
use vortex_array::arrow::FromArrowArray;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;

use super::ArrayFixture;

const SCALE_FACTOR: f64 = 0.01;

fn collect_batches_as_vortex(iter: impl RecordBatchIterator) -> VortexResult<ArrayRef> {
    let batches: Vec<RecordBatch> = iter.collect();
    Ok(ChunkedArray::from_iter(
        batches
            .into_iter()
            .map(|batch| ArrayRef::from_arrow(batch, false))
            .collect::<VortexResult<Vec<_>>>()?,
    )
    .into_array())
}

struct TpchLineitemFixture;

impl ArrayFixture for TpchLineitemFixture {
    fn name(&self) -> &str {
        "tpch_lineitem.vortex"
    }

    fn description(&self) -> &str {
        "TPC-H lineitem table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![
            Struct::ID,
            Primitive::ID,
            Decimal::ID,
            VarBinView::ID,
            Extension::ID,
        ]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let generator = LineItemGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::LineItemArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}

struct TpchOrdersFixture;

impl ArrayFixture for TpchOrdersFixture {
    fn name(&self) -> &str {
        "tpch_orders.vortex"
    }

    fn description(&self) -> &str {
        "TPC-H orders table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![
            Struct::ID,
            Primitive::ID,
            Decimal::ID,
            VarBinView::ID,
            Extension::ID,
        ]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let generator = OrderGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::OrderArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}

pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    vec![Box::new(TpchLineitemFixture), Box::new(TpchOrdersFixture)]
}
