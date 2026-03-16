// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use arrow_array::RecordBatch;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen_arrow::RecordBatchIterator;
use vortex::layout::LayoutId;
use vortex_array::ArrayRef;
use vortex_array::arrow::FromArrowArray;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;

use super::ExpectedEncoding;
use super::Fixture;

const SCALE_FACTOR: f64 = 0.01;

fn collect_batches_as_vortex(iter: impl RecordBatchIterator) -> VortexResult<Vec<ArrayRef>> {
    let batches: Vec<RecordBatch> = iter.collect();
    batches
        .into_iter()
        .map(|batch| ArrayRef::from_arrow(batch, false))
        .collect()
}

pub struct TpchLineitemFixture;

impl Fixture for TpchLineitemFixture {
    fn name(&self) -> &str {
        "tpch_lineitem.vortex"
    }

    fn description(&self) -> &str {
        "TPC-H lineitem table at scale factor 0.01"
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

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let generator = LineItemGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::LineItemArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}

pub struct TpchOrdersFixture;

impl Fixture for TpchOrdersFixture {
    fn name(&self) -> &str {
        "tpch_orders.vortex"
    }

    fn description(&self) -> &str {
        "TPC-H orders table at scale factor 0.01"
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

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let generator = OrderGenerator::new(SCALE_FACTOR, 1, 1);
        let arrow_iter = tpchgen_arrow::OrderArrow::new(generator).with_batch_size(65_536);
        collect_batches_as_vortex(arrow_iter)
    }
}
