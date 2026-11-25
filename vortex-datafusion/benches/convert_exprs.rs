// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converting nested DataFusion physical expressions to Vortex expressions.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::Literal;
use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_datafusion::convert::DefaultExpressionConverter;
use vortex_datafusion::convert::ExpressionConverter;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const COLUMNS: usize = 64;

fn schema() -> Schema {
    Schema::new(
        (0..COLUMNS)
            .map(|idx| Field::new(format!("c{idx}"), DataType::Int64, true))
            .collect::<Vec<_>>(),
    )
}

fn column(idx: usize) -> Arc<dyn PhysicalExpr> {
    let idx = idx % COLUMNS;
    Arc::new(Column::new(&format!("c{idx}"), idx))
}

fn literal(value: i64) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int64(Some(value))))
}

fn binary(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, op, right))
}

/// `c0 = 0 OR c1 = 1 OR ...`, nested to the left as SQL parses it.
fn or_chain(n: usize) -> Arc<dyn PhysicalExpr> {
    (0..n)
        .map(|idx| binary(column(idx), Operator::Eq, literal(idx as i64)))
        .reduce(|chain, comparison| binary(chain, Operator::Or, comparison))
        .unwrap()
}

/// `c0 + c1 + ... > 0`, nested to the left as SQL parses it.
fn sum_chain(n: usize) -> Arc<dyn PhysicalExpr> {
    let sum = (0..n)
        .map(column)
        .reduce(|sum, column| binary(sum, Operator::Plus, column))
        .unwrap();
    binary(sum, Operator::Gt, literal(0))
}

fn bench_convert(bencher: Bencher, expr: Arc<dyn PhysicalExpr>) {
    let schema = schema();
    let converter = DefaultExpressionConverter::default();
    assert!(converter.try_convert(&expr, &schema).unwrap().is_some());
    bencher.bench(|| converter.try_convert(&expr, &schema).unwrap());
}

#[divan::bench(args = [8, 64, 512])]
fn convert_or_chain(bencher: Bencher, n: usize) {
    bench_convert(bencher, or_chain(n));
}

#[divan::bench(args = [8, 64, 512])]
fn convert_sum_chain(bencher: Bencher, n: usize) {
    bench_convert(bencher, sum_chain(n));
}
