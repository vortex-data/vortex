// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::future::pending;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::pruning::PrunableStatistics;
use datafusion_common::stats::Precision;
use datafusion_datasource::PartitionedFile;
use datafusion_expr::ColumnarValue;
use datafusion_expr::Operator;
use datafusion_expr::ScalarUDF;
use datafusion_functions::core::coalesce;
use datafusion_functions::datetime::now;
use datafusion_functions::math::abs;
use datafusion_functions::math::random;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::NotExpr;
use datafusion_physical_plan::joins::HashExpr;
use datafusion_physical_plan::joins::SeededRandomState;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::build_pruning_predicate;
use parking_lot::RwLock;
use rstest::rstest;
use tokio::sync::Barrier;
use vortex::error::VortexExpect;

use super::CacheKind;
use super::FilePruner;
use super::FilePruningCache;
use super::prune_file;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
}

fn predicate() -> PhysicalExprRef {
    greater_than(10)
}

fn greater_than(value: i32) -> PhysicalExprRef {
    Arc::new(BinaryExpr::new(
        Arc::new(Column::new("a", 0)),
        Operator::Gt,
        Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
    ))
}

fn file(schema: &Schema, max: i32) -> PartitionedFile {
    file_range(schema, 0, max)
}

fn file_range(schema: &Schema, min: i32, max: i32) -> PartitionedFile {
    let mut statistics = Statistics::new_unknown(schema);
    statistics.column_statistics[0].min_value = Precision::Exact(ScalarValue::Int32(Some(min)));
    statistics.column_statistics[0].max_value = Precision::Exact(ScalarValue::Int32(Some(max)));
    PartitionedFile::new("file.vortex", 100).with_statistics(Arc::new(statistics))
}

#[derive(Debug)]
struct CustomSnapshotExpr {
    state: RwLock<(u64, PhysicalExprRef)>,
    versioned: bool,
}

impl CustomSnapshotExpr {
    fn update(&self, expression: PhysicalExprRef) {
        let mut state = self.state.write();
        state.0 += 1;
        state.1 = expression;
    }
}

impl PartialEq for CustomSnapshotExpr {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for CustomSnapshotExpr {}

impl Hash for CustomSnapshotExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}

impl fmt::Display for CustomSnapshotExpr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("custom_snapshot")
    }
}

impl PhysicalExpr for CustomSnapshotExpr {
    fn data_type(&self, _schema: &Schema) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _schema: &Schema) -> DFResult<bool> {
        Ok(false)
    }

    fn fmt_sql(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }

    fn evaluate(&self, batch: &arrow_array::RecordBatch) -> DFResult<ColumnarValue> {
        self.state.read().1.evaluate(batch)
    }

    fn children(&self) -> Vec<&PhysicalExprRef> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<PhysicalExprRef>,
    ) -> DFResult<PhysicalExprRef> {
        assert!(children.is_empty());
        Ok(self)
    }

    fn snapshot(&self) -> DFResult<Option<PhysicalExprRef>> {
        Ok(Some(Arc::clone(&self.state.read().1)))
    }

    fn snapshot_generation(&self) -> u64 {
        if self.versioned {
            self.state.read().0
        } else {
            0
        }
    }
}

#[rstest]
#[case::immutable(abs(), true, CacheKind::Static)]
#[case::stable(now(&Default::default()), false, CacheKind::Uncached)]
#[case::volatile(random(), false, CacheKind::Uncached)]
fn scalar_function_volatility_controls_sharing(
    #[case] function: Arc<ScalarUDF>,
    #[case] column_argument: bool,
    #[case] expected: CacheKind,
) -> DFResult<()> {
    let arguments = if column_argument {
        vec![Arc::new(Column::new("a", 0)) as PhysicalExprRef]
    } else {
        Vec::new()
    };
    let expression = Arc::new(ScalarFunctionExpr::try_new(
        function,
        arguments,
        &schema(),
        Arc::default(),
    )?) as PhysicalExprRef;
    assert_eq!(CacheKind::classify(&expression), expected);
    Ok(())
}

#[rstest]
fn immutable_wrappers_still_check_their_children(
    #[values(false, true)] custom_child: bool,
) -> DFResult<()> {
    let child = if custom_child {
        Arc::new(CustomSnapshotExpr {
            state: RwLock::new((0, predicate())),
            versioned: false,
        }) as PhysicalExprRef
    } else {
        Arc::new(Column::new("a", 0)) as PhysicalExprRef
    };
    let function = Arc::new(ScalarFunctionExpr::try_new(
        coalesce(),
        vec![Arc::clone(&child), Arc::clone(&child)],
        &schema(),
        Arc::default(),
    )?) as PhysicalExprRef;
    let hash = Arc::new(HashExpr::new(
        vec![child],
        SeededRandomState::with_seed(0),
        "hash".to_owned(),
    )) as PhysicalExprRef;
    let expected = if custom_child {
        CacheKind::Uncached
    } else {
        CacheKind::Static
    };
    assert_eq!(CacheKind::classify(&function), expected);
    assert_eq!(CacheKind::classify(&hash), expected);
    Ok(())
}

async fn initial_file_check(
    cache: &Arc<FilePruningCache>,
    expression: &PhysicalExprRef,
    schema: &SchemaRef,
    file: &PartitionedFile,
) -> DFResult<bool> {
    let errors = Count::new();
    if let Some(pruned) = cache.should_prune(expression, schema, file, &errors) {
        return Ok(pruned);
    }
    FilePruner::try_new(
        Arc::clone(expression),
        schema,
        file,
        errors,
        Some(Arc::clone(cache)),
    )
    .vortex_expect("file statistics exist")
    .should_prune_initial()
    .await
}

#[rstest]
#[case::cached_none(true, 10, 5, false, true)]
#[case::tightening(false, 30, 20, false, true)]
#[case::broader_snapshot(false, -1, 5, true, false)]
#[tokio::test]
async fn custom_snapshots_are_refreshed_for_each_file(
    #[case] initially_true: bool,
    #[case] updated_bound: i32,
    #[case] maximum: i32,
    #[case] initial_pruned: bool,
    #[case] updated_pruned: bool,
    #[values(false, true)] versioned: bool,
    #[values(false, true)] with_builtin_dynamic: bool,
) -> DFResult<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let initial = if initially_true {
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))) as PhysicalExprRef
    } else {
        predicate()
    };
    let custom = Arc::new(CustomSnapshotExpr {
        state: RwLock::new((1, initial)),
        versioned,
    });
    let custom_expression = Arc::clone(&custom) as PhysicalExprRef;
    let expression = if with_builtin_dynamic {
        Arc::new(BinaryExpr::new(
            custom_expression,
            Operator::And,
            Arc::new(DynamicFilterPhysicalExpr::new(
                vec![Arc::new(Column::new("a", 0))],
                Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
            )),
        )) as PhysicalExprRef
    } else {
        custom_expression
    };
    let first = file(&schema, maximum);
    let mut second = first.clone();
    second.object_meta.location = "second.vortex".into();
    assert_eq!(
        initial_file_check(&cache, &expression, &schema, &first).await?,
        initial_pruned
    );
    custom.update(greater_than(updated_bound));
    assert_eq!(
        initial_file_check(&cache, &expression, &schema, &second).await?,
        updated_pruned
    );
    Ok(())
}

#[rstest]
#[case::already_rejected(5, true, true)]
#[case::newly_rejected(20, false, true)]
#[case::still_matches(40, false, false)]
#[tokio::test]
async fn older_dynamic_snapshots_are_conservative_under_refinement(
    #[case] maximum: i32,
    #[case] initial_pruned: bool,
    #[case] refined_pruned: bool,
    #[values(false, true)] hidden_from_start: bool,
) -> DFResult<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let inner = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Literal::new(ScalarValue::Int32(Some(0))))],
        Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
    ));
    let body = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("a", 0)),
        Operator::Gt,
        Arc::clone(&inner) as PhysicalExprRef,
    )) as PhysicalExprRef;
    let outer = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        if hidden_from_start {
            Arc::clone(&body)
        } else {
            predicate()
        },
    ));
    let expression = Arc::clone(&outer) as PhysicalExprRef;
    let file = file(&schema, maximum);
    assert_eq!(
        initial_file_check(&cache, &expression, &schema, &file).await?,
        initial_pruned
    );
    if !hidden_from_start {
        outer.update(Arc::clone(&body))?;
        assert_eq!(
            initial_file_check(&cache, &expression, &schema, &file).await?,
            initial_pruned
        );
    }
    inner.update(Arc::new(Literal::new(ScalarValue::Int32(Some(30)))))?;
    assert_eq!(
        initial_file_check(&cache, &expression, &schema, &file).await?,
        initial_pruned
    );
    // Publishing the refined body advances the outer generation and restores precise pruning.
    outer.update(body)?;
    assert_eq!(
        initial_file_check(&cache, &expression, &schema, &file).await?,
        refined_pruned
    );
    Ok(())
}

#[tokio::test]
async fn negated_dynamic_filter_uses_per_file_snapshots() -> DFResult<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let inner = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Literal::new(ScalarValue::Int32(Some(0))))],
        Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
    ));
    let outer = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            Arc::clone(&inner) as PhysicalExprRef,
        )),
    ));
    let expression = Arc::new(NotExpr::new(outer)) as PhysicalExprRef;
    let file = file_range(&schema, 15, 20);
    assert!(matches!(
        FilePruner::try_new(
            Arc::clone(&expression),
            &schema,
            &file,
            Count::new(),
            Some(Arc::clone(&cache)),
        ),
        Some(FilePruner::DataFusion(_))
    ));
    assert!(initial_file_check(&cache, &expression, &schema, &file).await?);
    inner.update(Arc::new(Literal::new(ScalarValue::Int32(Some(30)))))?;
    assert!(!initial_file_check(&cache, &expression, &schema, &file).await?);
    Ok(())
}

#[rstest]
#[case::positive_and(Operator::And, true, CacheKind::Dynamic)]
#[case::positive_or(Operator::Or, true, CacheKind::Dynamic)]
#[case::negative_and(Operator::And, false, CacheKind::Uncached)]
#[case::negative_or(Operator::Or, false, CacheKind::Uncached)]
#[case::comparison(Operator::Eq, true, CacheKind::Uncached)]
fn dynamic_filter_composition_preserves_refinement(
    #[case] operator: Operator,
    #[case] positive: bool,
    #[case] expected: CacheKind,
) {
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    )) as PhysicalExprRef;
    let expression = Arc::new(BinaryExpr::new(
        filter,
        operator,
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
    )) as PhysicalExprRef;
    let expression = if positive {
        expression
    } else {
        Arc::new(NotExpr::new(expression)) as PhysicalExprRef
    };
    assert_eq!(CacheKind::classify(&expression), expected);
}

#[test]
fn shared_predicate_evaluates_each_file_statistics() {
    let cache = FilePruningCache::default();
    let schema = schema();
    let predicate = predicate();
    let errors = Count::new();
    for (max, expected) in [(5, true), (20, false), (5, true)] {
        assert_eq!(
            cache.should_prune(&predicate, &schema, &file(&schema, max), &errors),
            Some(expected)
        );
    }
    assert_eq!(errors.value(), 0);
}

#[test]
fn missing_statistics_do_not_disable_later_pruning() {
    let cache = FilePruningCache::default();
    let schema = schema();
    let predicate = predicate();
    let errors = Count::new();
    let unknown = PartitionedFile::new("unknown.vortex", 100)
        .with_statistics(Arc::new(Statistics::new_unknown(&schema)));
    for file in [PartitionedFile::new("absent.vortex", 100), unknown] {
        assert_eq!(
            cache.should_prune(&predicate, &schema, &file, &errors),
            Some(false)
        );
        assert!(cache.predicate.get().is_none());
    }
    assert_eq!(
        cache.should_prune(&predicate, &schema, &file(&schema, 5), &errors),
        Some(true)
    );
}

#[test]
fn dynamic_predicates_keep_their_file_pruners() -> anyhow::Result<()> {
    let cache = FilePruningCache::default();
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
    ));
    let expression = Arc::clone(&dynamic) as PhysicalExprRef;
    let errors = Count::new();
    assert_eq!(
        cache.should_prune(&expression, &schema, &file(&schema, 5), &errors),
        None
    );
    dynamic.update(predicate())?;
    dynamic.mark_complete();
    assert_eq!(
        cache.should_prune(&expression, &schema, &file(&schema, 5), &errors),
        None
    );
    assert!(cache.predicate.get().is_none());
    Ok(())
}

#[test]
fn dynamic_pruners_share_compilation_and_watch_each_file() -> anyhow::Result<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
    ));
    let expression = Arc::clone(&dynamic) as PhysicalExprRef;
    let errors = Count::new();
    let mut small = FilePruner::try_new(
        Arc::clone(&expression),
        &schema,
        &file(&schema, 5),
        errors.clone(),
        Some(Arc::clone(&cache)),
    )
    .vortex_expect("file statistics exist");
    let mut large = FilePruner::try_new(
        Arc::clone(&expression),
        &schema,
        &file(&schema, 20),
        errors.clone(),
        Some(Arc::clone(&cache)),
    )
    .vortex_expect("file statistics exist");
    assert!(small.is_watching());
    assert!(!small.should_prune()?);
    assert!(!large.should_prune()?);
    assert!(
        cache
            .dynamic_predicate(&expression, || panic!("cached true"))
            .is_none()
    );

    dynamic.update(predicate())?;
    assert!(small.should_prune()?);
    assert!(!large.should_prune()?);
    let compiled = cache
        .dynamic_predicate(&expression, || panic!("generation already compiled"))
        .vortex_expect("bounds are prunable");
    assert!(Arc::ptr_eq(
        &compiled,
        &cache
            .dynamic_predicate(&expression, || panic!("shared compiled predicate"))
            .vortex_expect("bounds are prunable")
    ));

    dynamic.update(greater_than(30))?;
    assert!(small.should_prune()?);
    assert!(large.should_prune()?);
    assert_eq!(errors.value(), 0);
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn completed_dynamic_filters_are_checked(#[case] initially_complete: bool) -> anyhow::Result<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
    ));
    if initially_complete {
        dynamic.update(predicate())?;
        dynamic.mark_complete();
    }
    let expression = Arc::clone(&dynamic) as PhysicalExprRef;
    let unknown = PartitionedFile::new("unknown.vortex", 100)
        .with_statistics(Arc::new(Statistics::new_unknown(&schema)));
    let mut unknown = FilePruner::try_new(
        Arc::clone(&expression),
        &schema,
        &unknown,
        Count::new(),
        Some(Arc::clone(&cache)),
    )
    .vortex_expect("dynamic predicates accept unknown statistics");
    let mut pruner = FilePruner::try_new(
        expression,
        &schema,
        &file(&schema, 5),
        Count::new(),
        Some(cache),
    )
    .vortex_expect("file statistics exist");
    assert_eq!(pruner.is_watching(), !initially_complete);
    if !initially_complete {
        assert!(!pruner.should_prune()?);
        dynamic.update(predicate())?;
        dynamic.mark_complete();
    }
    assert!(pruner.should_prune()?);
    assert!(!pruner.should_prune()?);
    assert!(!unknown.should_prune()?);
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn changing_generation_during_compilation_is_not_published(
    #[case] publish_newer: bool,
) -> anyhow::Result<()> {
    let cache = FilePruningCache::default();
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    ));
    let expression = Arc::clone(&dynamic) as PhysicalExprRef;
    let errors = Count::new();
    let previous = cache
        .dynamic_predicate(&expression, || {
            let compiled = build_pruning_predicate(Arc::clone(&expression), &schema, &errors);
            dynamic.update(greater_than(30)).expect("valid update");
            if publish_newer {
                drop(cache.dynamic_predicate(&expression, || {
                    build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
                }));
            }
            compiled
        })
        .vortex_expect("bounds are prunable");
    assert_eq!(
        cache
            .dynamic_predicate
            .read()
            .as_ref()
            .and_then(|entry| entry.predicate.get())
            .is_some(),
        publish_newer
    );
    let current = cache
        .dynamic_predicate(&expression, || {
            build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
        })
        .vortex_expect("updated bounds are prunable");
    assert!(!Arc::ptr_eq(&previous, &current));
    let statistics = PrunableStatistics::new(
        vec![
            file(&schema, 20)
                .statistics
                .vortex_expect("file statistics"),
        ],
        schema,
    );
    assert!(!prune_file(&previous, &statistics, &errors));
    assert!(prune_file(&current, &statistics, &errors));
    Ok(())
}

#[tokio::test]
async fn concurrent_initial_checks_compile_once() {
    let cache = FilePruningCache::default();
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    ));
    let expression = dynamic as PhysicalExprRef;
    let errors = Count::new();
    let (first, second) = futures::join!(
        cache.initial_dynamic_predicate(&expression, || async {
            tokio::task::yield_now().await;
            build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
        }),
        cache.initial_dynamic_predicate(&expression, || async {
            panic!("the second file must await the first compilation")
        })
    );
    assert!(Arc::ptr_eq(
        &first.vortex_expect("bounds are prunable"),
        &second.vortex_expect("same compiled bounds")
    ));
}

#[tokio::test]
async fn canceled_initializer_allows_waiter_to_take_over() -> anyhow::Result<()> {
    let cache = FilePruningCache::default();
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    ));
    let expression = dynamic as PhysicalExprRef;
    let errors = Count::new();
    let mut initializer = Box::pin(cache.initial_dynamic_predicate(&expression, || async {
        pending::<()>().await;
        unreachable!("initializer must be canceled")
    }));
    assert!(futures::poll!(&mut initializer).is_pending());
    let mut waiter = Box::pin(cache.initial_dynamic_predicate(&expression, || async {
        build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
    }));
    assert!(futures::poll!(&mut waiter).is_pending());
    drop(initializer);
    let compiled = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await?
        .vortex_expect("the waiter compiles the bounds");
    let statistics = PrunableStatistics::new(
        vec![file(&schema, 5).statistics.vortex_expect("file statistics")],
        Arc::clone(&schema),
    );
    assert!(prune_file(&compiled, &statistics, &errors));
    let cached = cache
        .initial_dynamic_predicate(&expression, || async {
            panic!("the waiter's result must be published")
        })
        .await
        .vortex_expect("published bounds");
    assert!(Arc::ptr_eq(&compiled, &cached));
    assert_eq!(errors.value(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn initial_checks_share_compilation_across_workers() -> anyhow::Result<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let expression = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    )) as PhysicalExprRef;
    let builds = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(16));
    let tasks = (0..16)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let schema = Arc::clone(&schema);
            let expression = Arc::clone(&expression);
            let builds = Arc::clone(&builds);
            let start = Arc::clone(&start);
            tokio::spawn(async move {
                start.wait().await;
                cache
                    .initial_dynamic_predicate(&expression, || async {
                        builds.fetch_add(1, Ordering::Relaxed);
                        tokio::task::yield_now().await;
                        build_pruning_predicate(Arc::clone(&expression), &schema, &Count::new())
                    })
                    .await
                    .vortex_expect("bounds are prunable")
            })
        })
        .collect::<Vec<_>>();
    let mut compiled = Vec::with_capacity(tasks.len());
    for task in tasks {
        compiled.push(task.await?);
    }
    assert_eq!(builds.load(Ordering::Relaxed), 1);
    assert!(compiled.iter().all(|item| Arc::ptr_eq(item, &compiled[0])));
    Ok(())
}

#[tokio::test]
async fn update_and_completion_while_awaiting_remain_visible() -> anyhow::Result<()> {
    let cache = Arc::new(FilePruningCache::default());
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    ));
    let expression = Arc::clone(&dynamic) as PhysicalExprRef;
    let errors = Count::new();
    let mut pruner = FilePruner::try_new(
        Arc::clone(&expression),
        &schema,
        &file(&schema, 20),
        errors.clone(),
        Some(Arc::clone(&cache)),
    )
    .vortex_expect("file statistics exist");
    let (_, initial) = futures::join!(
        cache.initial_dynamic_predicate(&expression, || async {
            let previous = build_pruning_predicate(Arc::clone(&expression), &schema, &errors);
            tokio::task::yield_now().await;
            dynamic.update(greater_than(30)).expect("valid update");
            dynamic.mark_complete();
            previous
        }),
        pruner.should_prune_initial(),
    );
    assert!(!initial?);
    assert!(
        cache
            .dynamic_predicate
            .read()
            .as_ref()
            .vortex_expect("an old generation was requested")
            .predicate
            .get()
            .is_none()
    );
    assert!(pruner.should_prune()?);
    assert_eq!(errors.value(), 0);
    Ok(())
}

#[tokio::test]
async fn stream_check_does_not_wait_for_initial_compilation() {
    let cache = FilePruningCache::default();
    let schema = schema();
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("a", 0))],
        predicate(),
    ));
    let expression = dynamic as PhysicalExprRef;
    let errors = Count::new();
    let (initial, stream) = futures::join!(
        cache.initial_dynamic_predicate(&expression, || async {
            tokio::task::yield_now().await;
            build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
        }),
        async {
            let local = cache.dynamic_predicate(&expression, || {
                build_pruning_predicate(Arc::clone(&expression), &schema, &errors)
            });
            assert!(
                cache
                    .dynamic_predicate
                    .read()
                    .as_ref()
                    .vortex_expect("generation exists")
                    .predicate
                    .get()
                    .is_none()
            );
            local
        }
    );
    assert!(!Arc::ptr_eq(
        &initial.vortex_expect("initial bounds"),
        &stream.vortex_expect("local stream bounds")
    ));
}
