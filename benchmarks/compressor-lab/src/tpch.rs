// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-H tables generated in-process. Generation is deterministic for a scale factor, so a table
//! is identified by its name and scale factor alone.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::bail;
use arrow_array::RecordBatch;
use parking_lot::Mutex;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::NationGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::PartSuppGenerator;
use tpchgen::generators::RegionGenerator;
use tpchgen::generators::SupplierGenerator;

/// Rows per generated batch; tables are concatenated per column when chunked.
const BATCH: usize = 65_536;

type Cache = BTreeMap<(String, u64), Arc<Vec<RecordBatch>>>;

static CACHE: Mutex<Cache> = Mutex::new(BTreeMap::new());

/// The batches of a TPC-H table, generated once per process.
pub fn table(name: &str, scale_factor: f64) -> anyhow::Result<Arc<Vec<RecordBatch>>> {
    let key = (name.to_string(), scale_factor.to_bits());
    if let Some(batches) = CACHE.lock().get(&key) {
        return Ok(Arc::clone(batches));
    }
    let sf = scale_factor;
    let batches: Vec<RecordBatch> = match name {
        "lineitem" => tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "orders" => tpchgen_arrow::OrderArrow::new(OrderGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "partsupp" => tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "customer" => tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "part" => tpchgen_arrow::PartArrow::new(PartGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "supplier" => tpchgen_arrow::SupplierArrow::new(SupplierGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "nation" => tpchgen_arrow::NationArrow::new(NationGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        "region" => tpchgen_arrow::RegionArrow::new(RegionGenerator::new(sf, 1, 1))
            .with_batch_size(BATCH)
            .collect(),
        other => bail!("unknown TPC-H table `{other}`"),
    };
    let batches = Arc::new(batches);
    CACHE.lock().insert(key, Arc::clone(&batches));
    Ok(batches)
}
