// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VBench Q1 (cosine top-k) and Q4 (price-filtered cosine top-k).

use std::env;
use std::sync::Arc;

use anyhow::ensure;
use criterion::BenchmarkId;
use criterion::Criterion;
use datafusion::arrow::array::AsArray;
use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::types::UInt64Type;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;
use lance::Dataset;
use lance_bench::vbench::enable_ann;
use lance_bench::vbench::register_lance;
use lance_bench::vbench::register_vortex;
use tokio::runtime::Runtime;

use super::Data;
use super::K;
use super::NPROBE;
use super::VortexSide;
use super::exact_cosine;
use super::overlap;
use super::price;

struct Workload {
    sql: String,
    min_price: Option<i64>,
}

impl Workload {
    fn new(table: &str, min_price: Option<i64>) -> Self {
        let template = if min_price.is_some() {
            include_str!("../../queries/vbench/q4.sql")
        } else {
            include_str!("../../queries/vbench/q1.sql")
        };
        Self {
            sql: template.replace("FROM recipes", &format!("FROM {table}")),
            min_price,
        }
    }
}

fn parameter(vector: &[f32]) -> ScalarValue {
    let items: Vec<_> = vector.iter().copied().map(ScalarValue::from).collect();
    ScalarValue::List(ScalarValue::new_list(&items, &DataType::Float32, false))
}

async fn query(
    ctx: &SessionContext,
    workload: &Workload,
    vector: &ScalarValue,
    check_plan: Option<&str>,
) -> anyhow::Result<Vec<u64>> {
    let mut params = vec![vector.clone()];
    params.extend(workload.min_price.map(ScalarValue::from));
    let frame = ctx.sql(&workload.sql).await?.with_param_values(params)?;
    let plan = frame.create_physical_plan().await?;
    if let Some(expected) = check_plan {
        let text = displayable(plan.as_ref()).indent(true).to_string();
        // In verbose mode LanceNearestExec displays the native physical plan it executes.
        ensure!(
            text.contains(expected) && (expected != "LanceNearestExec" || text.contains("ANNIvf")),
            "expected {expected} in the SQL plan:\n{}\n{text}",
            frame.logical_plan().display_indent()
        );
    }
    let batches = collect(plan, ctx.task_ctx()).await?;
    let mut ids = Vec::new();
    let mut previous = f64::NEG_INFINITY;
    for batch in batches {
        let id = batch.column(0).as_primitive::<UInt64Type>();
        let distance = batch.column(1).as_primitive::<Float64Type>();
        for (id, distance) in id.values().iter().zip(distance.values()) {
            if check_plan.is_some() {
                ensure!(*distance >= previous, "unordered cosine distance");
                previous = *distance;
            }
            ids.push(*id);
        }
    }
    Ok(ids)
}

fn probes(name: &str) -> anyhow::Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(NPROBE),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn benchmark(
    c: &mut Criterion,
    runtime: &Runtime,
    data: &Data,
    vortex: Arc<VortexSide>,
    lance: Arc<Dataset>,
) {
    let ctx = SessionContext::new();
    let vortex_probes = probes("VBENCH_VORTEX_NPROBES").unwrap();
    let lance_probes = probes("VBENCH_LANCE_NPROBES").unwrap();
    eprintln!("VBench nprobes: Vortex {vortex_probes}, Lance {lance_probes}");
    register_vortex(
        &ctx,
        "vortex",
        vortex.data.clone(),
        &vortex.ivf,
        vortex_probes,
    )
    .unwrap();
    register_lance(&ctx, "lance", Arc::clone(&lance), lance_probes).unwrap();

    // Exercise the exact path before opting these same registered tables into ANN.
    let first = parameter(&data.query);
    for floor in [None, Some(90)] {
        let expected = exact_cosine(data, &data.query, floor);
        for (table, node) in [("vortex", "VortexScanExec"), ("lance", "LanceScanExec")] {
            let ids = runtime
                .block_on(query(
                    &ctx,
                    &Workload::new(table, floor),
                    &first,
                    Some(node),
                ))
                .unwrap();
            assert_eq!(overlap(&ids, &expected), expected.len());
        }
    }
    enable_ann(&ctx);
    for (table, node) in [("vortex", "VortexIvfExec"), ("lance", "LanceNearestExec")] {
        let ids = runtime
            .block_on(query(
                &ctx,
                &Workload::new(table, Some(100)),
                &first,
                Some(node),
            ))
            .unwrap();
        assert!(ids.is_empty(), "price >= 100 must match no rows");
    }

    let mut group = c.benchmark_group("vbench_sql");
    let queries = data.vector_queries();
    let parameters: Vec<_> = queries.iter().map(|q| parameter(q)).collect();
    for (case, floor) in [("q1", None), ("q4_price_90", Some(90))] {
        let exact: Vec<_> = queries
            .iter()
            .map(|q| exact_cosine(data, q, floor))
            .collect();
        for (table, node) in [("vortex", "VortexIvfExec"), ("lance", "LanceNearestExec")] {
            let statement = Workload::new(table, floor);
            let mut found = 0;
            for (vector, expected) in parameters.iter().zip(&exact) {
                let ids = runtime
                    .block_on(query(&ctx, &statement, vector, Some(node)))
                    .unwrap();
                assert!(ids.len() <= expected.len());
                assert!(
                    ids.iter()
                        .all(|&id| floor.is_none_or(|p| price(id as usize) >= p))
                );
                let mut distinct = ids.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert_eq!(distinct.len(), ids.len());
                found += overlap(&ids, expected);
            }
            eprintln!(
                "{case} {table} recall@{K}: {:.3}",
                found as f64 / (queries.len() * K) as f64
            );
            group.bench_function(
                BenchmarkId::new(format!("{case}/{table}"), data.scale.label),
                |b| {
                    let mut vectors = parameters.iter().cycle();
                    b.iter(|| {
                        let vector = vectors.next().unwrap();
                        runtime
                            .block_on(query(&ctx, &statement, vector, None))
                            .unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}
