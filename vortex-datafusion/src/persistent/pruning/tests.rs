// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_datasource::PartitionedFile;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::expressions::lit;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::metrics::MetricBuilder;
use rstest::rstest;

use super::FilePruners;
use super::Pruner;
use super::PruningState;

#[test]
fn dynamic_pruning_is_shared_and_remains_terminal() -> DFResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let mut stats = Statistics::new_unknown(&schema);
    stats.column_statistics[0].min_value = Precision::Exact(ScalarValue::Int32(Some(1)));
    stats.column_statistics[0].max_value = Precision::Exact(ScalarValue::Int32(Some(5)));
    let file = PartitionedFile::new("file.vortex", 100).with_statistics(Arc::new(stats));
    let column = col("a", &schema)?;
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&column)],
        lit(true),
    ));
    let predicate: PhysicalExprRef = dynamic.clone();
    let registry = FilePruners::default();
    let errors = MetricBuilder::new(&ExecutionPlanMetricsSet::new()).global_counter("errors");
    let mut first = Pruner::try_new(
        Some(&registry),
        Arc::clone(&predicate),
        &schema,
        &file,
        errors.clone(),
    )
    .expect("pruner");
    let mut second =
        Pruner::try_new(Some(&registry), predicate, &schema, &file, errors).expect("pruner");
    assert!(!first.should_prune()?);
    assert!(!second.should_prune()?);
    dynamic.update(Arc::new(BinaryExpr::new(column, Operator::Gt, lit(10_i32))))?;
    assert!(first.should_prune()?);
    assert!(second.should_prune()?);
    assert!(first.should_prune()?);
    Ok(())
}

#[rstest]
#[case("same", true)]
#[case("version", false)]
#[case("statistics", false)]
#[case("schema", false)]
#[case("predicate", false)]
fn sharing_requires_matching_identities(
    #[case] change: &str,
    #[case] shared: bool,
) -> DFResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let mut stats = Statistics::new_unknown(&schema);
    stats.column_statistics[0].min_value = Precision::Exact(ScalarValue::Int32(Some(1)));
    let file = PartitionedFile::new("file.vortex", 100).with_statistics(Arc::new(stats));
    let predicate = lit(false);
    let registry = FilePruners::default();
    let errors = MetricBuilder::new(&ExecutionPlanMetricsSet::new()).global_counter("errors");
    let first = Pruner::try_new(
        Some(&registry),
        Arc::clone(&predicate),
        &schema,
        &file,
        errors.clone(),
    )
    .expect("pruner");
    let mut other_file = file.clone();
    let mut other_schema = Arc::clone(&schema);
    let mut other_predicate = Arc::clone(&predicate);
    match change {
        "version" => other_file.object_meta.version = Some("new".to_owned()),
        "statistics" => {
            other_file.statistics = file
                .statistics
                .as_ref()
                .map(|stats| Arc::new(stats.as_ref().clone()))
        }
        "schema" => other_schema = Arc::new(schema.as_ref().clone()),
        "predicate" => other_predicate = lit(true),
        "same" => {}
        _ => unreachable!(),
    }
    let second = Pruner::try_new(
        Some(&registry),
        other_predicate,
        &other_schema,
        &other_file,
        errors,
    )
    .expect("pruner");
    let (PruningState::Shared(first), PruningState::Shared(second)) = (&first.0, &second.0) else {
        panic!("expected shared handles");
    };
    assert_eq!(Arc::ptr_eq(first, second), shared);
    Ok(())
}

#[test]
fn registry_does_not_keep_file_state_alive() -> DFResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let mut stats = Statistics::new_unknown(&schema);
    stats.column_statistics[0].min_value = Precision::Exact(ScalarValue::Int32(Some(1)));
    let file = PartitionedFile::new("file.vortex", 100).with_statistics(Arc::new(stats));
    let registry = FilePruners::default();
    let errors = MetricBuilder::new(&ExecutionPlanMetricsSet::new()).global_counter("errors");
    let pruner =
        Pruner::try_new(Some(&registry), lit(false), &schema, &file, errors).expect("pruner");
    let weak = match &pruner.0 {
        PruningState::Shared(state) => Arc::downgrade(state),
        PruningState::Owned(_) => panic!("expected shared handle"),
    };
    assert!(weak.upgrade().is_some());
    drop(pruner);
    assert!(weak.upgrade().is_none());
    assert!(
        registry
            .entries
            .iter()
            .all(|entry| entry.value().upgrade().is_none())
    );
    Ok(())
}
