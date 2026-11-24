// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx-lab probe`: site features and quality measures for every column of a TPC-H table, any
//! dtype, outside a plan. For checking the features and measures on real data quickly.

use std::io::Write;

use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::machine::TimingEnv;
use crate::measure;
use crate::measure::CandidateMeasures;
use crate::measure::TimingConfig;
use crate::measure::Workload;
use crate::site_features;
use crate::site_features::SiteFeatureRow;

/// One column's probe result.
#[derive(Debug, Serialize)]
pub struct ProbeRow {
    /// The table.
    pub table: String,
    /// The column.
    pub column: String,
    /// The dtype.
    pub dtype: String,
    /// Rows probed.
    pub rows: usize,
    /// Canonical bytes.
    pub canonical_bytes: u64,
    /// Production-compressed bytes.
    pub compressed_bytes: u64,
    /// The production encoding tree.
    pub tree: String,
    /// Features per distinct site.
    pub sites: Vec<SiteFeatureRow>,
    /// Measures of the canonical array and the production encoding.
    pub measures: Vec<CandidateMeasures>,
}

/// Probes the first `rows` rows of every column of a TPC-H table, writing one JSON line per
/// column to `out` and a summary line per column to stderr.
pub fn probe_tpch(
    table: &str,
    scale_factor: f64,
    rows: usize,
    config: &TimingConfig,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    let env = TimingEnv::current();
    if !env.warnings().is_empty() {
        eprintln!("timing conditions: {}", env.warnings().join("; "));
    }
    let batches = crate::tpch::table(table, scale_factor)?;
    let first = batches
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty table"))?;
    let schema = first.schema();
    let production = BtrBlocksCompressor::from_session(&session);
    for (index, field) in schema.fields().iter().enumerate() {
        let columns: Vec<&dyn arrow_array::Array> =
            batches.iter().map(|b| b.column(index).as_ref()).collect();
        let all = arrow_select::concat::concat(&columns)?;
        let arrow = all.slice(0, rows.min(all.len()));
        let array: ArrayRef = vortex_arrow::from_arrow_columnar(
            std::sync::Arc::clone(&arrow),
            arrow.len(),
            field.is_nullable(),
            &mut ctx,
        )?
        .execute::<Canonical>(&mut ctx)?
        .into_array();
        let sites = site_features::production_sites(&array, &session, &mut ctx)?;
        let encoded = production.compress(&array, &mut ctx)?;
        let workload = Workload::standard(&array, config.seed, &mut ctx)?;
        let measures = measure::measure(
            &array,
            &[("production".to_string(), encoded.clone())],
            &workload,
            config,
            &mut ctx,
        )?;
        let row = ProbeRow {
            table: table.to_string(),
            column: field.name().clone(),
            dtype: array.dtype().to_string(),
            rows: array.len(),
            canonical_bytes: array.nbytes(),
            compressed_bytes: encoded.nbytes(),
            tree: encoded
                .display_tree_encodings_only()
                .to_string()
                .lines()
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(" / "),
            sites,
            measures,
        };
        summarize(&row);
        serde_json::to_writer(&mut *out, &row)?;
        writeln!(out)?;
    }
    Ok(())
}

fn summarize(row: &ProbeRow) {
    let feature_ns: u64 = row
        .sites
        .iter()
        .flat_map(|s| s.features.costs.iter())
        .map(|c| c.ns)
        .sum();
    let features: usize = row.sites.iter().map(|s| s.features.features.len()).sum();
    let op = |name: &str, op: &str| {
        row.measures
            .iter()
            .find(|m| m.name == name)
            .and_then(|m| m.ops.iter().find(|o| o.op == op))
            .map(|o| {
                let mut s = o.samples_ns.clone();
                s.sort_by(f64::total_cmp);
                let median = s.get(s.len() / 2).copied().unwrap_or(f64::NAN);
                format!(
                    "{:.0}µs{}{}",
                    median / 1e3,
                    if o.correct { "" } else { " WRONG" },
                    if o.converged { "" } else { "~" }
                )
            })
            .unwrap_or_else(|| "-".into())
    };
    let wrong: usize = row
        .measures
        .iter()
        .flat_map(|m| m.ops.iter())
        .filter(|o| !o.correct)
        .count();
    eprintln!(
        "{:<18} {:<22} ratio {:>5.2}  sites {:>2}  features {:>4} in {:>6.1}ms  decode {} (canon {})  filter1% {}  take {}  eq {}  wrong {}",
        row.column,
        row.dtype,
        row.canonical_bytes as f64 / row.compressed_bytes.max(1) as f64,
        row.sites.len(),
        features,
        feature_ns as f64 / 1e6,
        op("production", "decode"),
        op(measure::CANONICAL, "decode"),
        op("production", "filter.random_1pct"),
        op("production", "take.random_1k"),
        op("production", "compare.eq_median"),
        wrong,
    );
}
