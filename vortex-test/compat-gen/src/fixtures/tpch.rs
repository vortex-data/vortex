use arrow_array::RecordBatch;
use tpchgen::generators::{LineItemGenerator, OrderGenerator};
use tpchgen_arrow::RecordBatchIterator;
use vortex_array::arrow::FromArrowArray;
use vortex_array::ArrayRef;

use super::Fixture;

const SCALE_FACTOR: f64 = 0.01;

fn collect_batches_as_vortex(iter: impl RecordBatchIterator) -> Vec<ArrayRef> {
    let batches: Vec<RecordBatch> = iter.collect();
    batches
        .into_iter()
        .map(|batch| ArrayRef::from_arrow(batch, false).expect("arrow conversion failed"))
        .collect()
}

pub struct TpchLineitemFixture;

impl Fixture for TpchLineitemFixture {
    fn name(&self) -> &str {
        "tpch_lineitem.vortex"
    }

    fn build(&self) -> Vec<ArrayRef> {
        let gen = LineItemGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::LineItemArrow::new(gen).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}

pub struct TpchOrdersFixture;

impl Fixture for TpchOrdersFixture {
    fn name(&self) -> &str {
        "tpch_orders.vortex"
    }

    fn build(&self) -> Vec<ArrayRef> {
        let gen = OrderGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::OrderArrow::new(gen).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}
