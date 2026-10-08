// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::dynamic;
use vortex_array::expr::eq;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt_eq;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::stats::PRUNING_STATS;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_layout::scan::metrics::ScanCounters;
use vortex_layout::scan::split_by::SplitBy;
use vortex_metrics::DefaultMetricsRegistry;
use vortex_metrics::MetricValue;
use vortex_metrics::MetricsRegistry;

use super::SESSION;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

async fn scan_file(values: [[i32; 2]; 4]) -> VortexResult<VortexFile> {
    let chunks = values
        .into_iter()
        .map(|[a, b]| StructArray::from_fields(&[("value", buffer![a, b].into_array())]))
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = chunks[0].dtype().clone();
    let array = ChunkedArray::try_new(
        chunks.into_iter().map(IntoArray::into_array),
        dtype,
    )?
    .into_array();
    let strategy = TableStrategy::new(
        Arc::new(FlatLayoutStrategy::default()),
        Arc::new(ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(2).unwrap(),
                ..Default::default()
            },
        )),
    );
    let mut buffer = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(Arc::new(strategy))
        .with_file_statistics(PRUNING_STATS.to_vec())
        .write(&mut buffer, array.to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buffer)
}

fn assert_registry(registry: &DefaultMetricsRegistry, expected: [u64; 4]) {
    for (idx, name) in [
        "scan.splits.considered",
        "scan.splits.pruned",
        "scan.splits.filtered",
        "scan.files.pruned",
    ]
    .into_iter()
    .enumerate()
    {
        let actual: Vec<_> = registry
            .snapshot()
            .into_iter()
            .filter(|metric| metric.name() == name)
            .map(|metric| match metric.value() {
                MetricValue::Counter(counter) => counter.value(),
                _ => panic!("expected a counter"),
            })
            .collect();
        assert_eq!(actual, vec![expected[idx]], "{name}");
    }
}

fn assert_execution(counters: &ScanCounters, expected: [u64; 4]) {
    let counts = counters.snapshot();
    assert_eq!(
        [
            counts.splits_considered,
            counts.splits_pruned,
            counts.splits_filtered,
            counts.files_pruned,
        ],
        expected,
    );
}

#[rstest]
#[case::statistics([[0, 1], [10, 11], [90, 91], [100, 101]], [4, 4, 0, 0])]
#[case::predicate([[0, 100], [10, 90], [20, 80], [30, 70]], [4, 0, 4, 0])]
#[tokio::test]
async fn test_scan_rejection_counters(
    #[case] values: [[i32; 2]; 4],
    #[case] expected: [u64; 4],
) -> VortexResult<()> {
    let file = scan_file(values).await?;
    let predicate = and(
        gt_eq(col("value"), lit(0i32)),
        and(eq(col("value"), lit(50i32)), lt_eq(col("value"), lit(101i32))),
    );
    assert!(!file.can_prune(&predicate)?);
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(predicate.bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .map(|_| -> VortexResult<()> { panic!("rejected splits must not reach the mapper") })
        .prepare()?;

    let mut previous = None;
    for execution in 1..=2 {
        let (stream, counters) = scan.execute_stream_with_counters(None)?;
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
        assert_execution(&counters, expected);
        assert_registry(&registry, expected.map(|count| count * execution));
        if let Some(previous) = &previous {
            assert!(!Arc::ptr_eq(previous, &counters));
            assert_execution(previous, expected);
        }
        previous = Some(counters);
    }
    Ok(())
}

#[tokio::test]
async fn test_scan_whole_file_rejection() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let predicate = and(eq(col("value"), lit(200i32)), eq(col("value"), lit(300i32)));
    assert!(file.can_prune(&predicate)?);
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(predicate.bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .prepare()?;
    let (tasks, first) = scan.execute_with_counters(None)?;
    assert!(tasks.is_empty());
    let (tasks, second) = scan.execute_with_counters(None)?;
    assert!(tasks.is_empty());
    assert_execution(&first, [0, 0, 0, 1]);
    assert_execution(&second, [0, 0, 0, 1]);
    assert_registry(&registry, [0, 0, 0, 2]);
    Ok(())
}

#[tokio::test]
async fn test_scan_lazy_whole_file_rejection() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let (stream, counters) = file
        .scan()?
        .with_filter(eq(col("value"), lit(200i32)).bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .into_stream_with_counters()?;
    assert!(registry.snapshot().is_empty());
    assert_execution(&counters, [0, 0, 0, 0]);
    futures::pin_mut!(stream);
    assert!(stream.next().await.is_none());
    assert_registry(&registry, [0, 0, 0, 1]);
    assert_execution(&counters, [0, 0, 0, 1]);
    Ok(())
}

#[rstest]
#[case::zero(0, 0, 0)]
#[case::partial(3, 3, 2)]
#[case::all(8, 8, 4)]
#[tokio::test]
async fn test_scan_limit_counters(
    #[case] limit: u64,
    #[case] rows: usize,
    #[case] considered: u64,
) -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_limit(limit)
        .with_metrics_registry(registry.clone())
        .prepare()?;
    let mut actual_rows = 0;
    let (tasks, counters) = scan.execute_with_counters(None)?;
    for task in tasks {
        actual_rows += task.await?.unwrap().len();
    }
    assert_eq!(actual_rows, rows);
    assert_registry(&registry, [considered, 0, 0, 0]);
    assert_execution(&counters, [considered, 0, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn test_scan_unpolled_tasks() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let (mut tasks, counters) = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(eq(col("value"), lit(50i32)).bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .build_with_counters()?;
    assert_eq!(tasks.len(), 4);
    assert_registry(&registry, [0, 0, 0, 0]);
    assert!(tasks.remove(0).await?.is_none());
    assert_execution(&counters, [1, 1, 0, 0]);
    assert_registry(&registry, [0, 0, 0, 0]);
    drop(tasks);
    assert_registry(&registry, [1, 1, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn test_scan_dynamic_file_rejection() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let bound = Arc::new(AtomicI32::new(50));
    let value = bound.clone();
    let predicate = dynamic(
        CompareOperator::Gt,
        move || Some(value.load(Ordering::Relaxed).into()),
        file.dtype().as_struct_fields().field_by_index(0).unwrap(),
        true,
        col("value"),
    );
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(and(predicate.clone(), predicate).bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .prepare()?;
    let (tasks, first) = scan.execute_with_counters(None)?;
    assert_eq!(tasks.len(), 4);
    bound.store(200, Ordering::Relaxed);
    for task in tasks {
        assert!(task.await?.is_none());
    }
    assert_registry(&registry, [4, 0, 0, 1]);
    assert_execution(&first, [4, 0, 0, 1]);

    let (tasks, second) = scan.execute_with_counters(None)?;
    assert!(tasks.is_empty());
    assert_registry(&registry, [4, 0, 0, 2]);
    assert_execution(&second, [0, 0, 0, 1]);
    bound.store(50, Ordering::Relaxed);
    let mut rows = 0;
    let (tasks, third) = scan.execute_with_counters(None)?;
    for task in tasks {
        if let Some(array) = task.await? {
            rows += array.len();
        }
    }
    assert_eq!(rows, 4);
    assert_registry(&registry, [8, 2, 0, 2]);
    assert_execution(&first, [4, 0, 0, 1]);
    assert_execution(&second, [0, 0, 0, 1]);
    assert_execution(&third, [4, 2, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn test_scan_registry_size_is_constant() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(eq(col("value"), lit(50i32)).bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .prepare()?;
    for _ in 0..100 {
        for task in scan.execute(None)? {
            assert!(task.await?.is_none());
        }
        assert_eq!(registry.snapshot().len(), 4);
    }
    assert_registry(&registry, [400, 400, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn test_scan_execution_counters_are_independent() -> VortexResult<()> {
    let file = scan_file([[0, 1], [10, 11], [90, 91], [100, 101]]).await?;
    let registry = Arc::new(DefaultMetricsRegistry::default());
    let scan = file
        .scan()?
        .with_split_by(SplitBy::RowCount(2))
        .with_filter(eq(col("value"), lit(50i32)).bind(file.dtype())?)
        .with_metrics_registry(registry.clone())
        .prepare()?;
    let (mut tasks, first) = scan.execute_with_counters(None)?;
    let (second_tasks, second) = scan.execute_with_counters(None)?;
    assert!(tasks.remove(0).await?.is_none());
    for task in second_tasks {
        assert!(task.await?.is_none());
    }
    assert_execution(&first, [1, 1, 0, 0]);
    assert_execution(&second, [4, 4, 0, 0]);
    assert_registry(&registry, [4, 4, 0, 0]);
    drop(tasks);
    assert_registry(&registry, [5, 5, 0, 0]);
    assert_execution(&first, [1, 1, 0, 0]);
    assert_execution(&second, [4, 4, 0, 0]);
    Ok(())
}
