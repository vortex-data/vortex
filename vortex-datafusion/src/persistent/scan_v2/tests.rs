// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use super::rebalance_batches;

#[rstest]
#[case(0, vec![3, 0, 10, 2])]
#[case(1, vec![1, 2, 8, 2, 1, 1])]
#[case(4, vec![4, 4, 4, 3])]
#[case(32, vec![15])]
#[tokio::test]
async fn rebalancing_preserves_rows_and_nulls(
    #[case] batch_size: usize,
    #[case] expected_lengths: Vec<usize>,
) -> VortexResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        true,
    )]));
    let values = (0..15)
        .map(|i| (i % 3 != 0).then_some(i))
        .collect::<Vec<_>>();
    let expected = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))])?;
    let input = vec![
        Ok(expected.slice(0, 3)),
        Ok(expected.slice(3, 0)),
        Ok(expected.slice(3, 10)),
        Ok(expected.slice(13, 2)),
    ];
    let batches: Vec<RecordBatch> = rebalance_batches(stream::iter(input).boxed(), batch_size)
        .try_collect()
        .await?;
    assert_eq!(
        batches
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        expected_lengths
    );
    let mut offset = 0;
    for batch in batches {
        assert_eq!(batch, expected.slice(offset, batch.num_rows()));
        offset += batch.num_rows();
    }
    assert_eq!(offset, expected.num_rows());
    Ok(())
}

#[tokio::test]
async fn rebalancing_preserves_zero_column_row_counts() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )?;
    let input = stream::iter(vec![Ok(batch.clone()), Ok(batch)]).boxed();
    let batches: Vec<RecordBatch> = rebalance_batches(input, 4).try_collect().await?;
    assert_eq!(
        batches
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![3, 3]
    );
    assert!(batches.iter().all(|batch| batch.num_columns() == 0));
    Ok(())
}

#[tokio::test]
async fn rebalancing_propagates_errors_with_a_pending_tail() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )?;
    let input = stream::iter(vec![Ok(batch), Err(vortex_err!("input failed"))]).boxed();
    let mut aligned = rebalance_batches(input, 4);
    let error = aligned
        .next()
        .await
        .expect("expected input error")
        .expect_err("input failed");
    assert!(error.to_string().contains("input failed"));
    assert!(aligned.next().await.is_none());
    Ok(())
}

#[tokio::test]
async fn rebalancing_avoids_a_one_row_final_batch() -> VortexResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]));
    let expected =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(0..17))])?;
    let batches: Vec<RecordBatch> =
        rebalance_batches(stream::iter(vec![Ok(expected.clone())]).boxed(), 8)
            .try_collect()
            .await?;
    assert_eq!(
        batches
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![6, 6, 5]
    );
    assert_eq!(batches[0], expected.slice(0, 6));
    assert_eq!(batches[1], expected.slice(6, 6));
    assert_eq!(batches[2], expected.slice(12, 5));
    Ok(())
}

#[test]
fn pending_rows_stay_below_two_target_batches_plus_two() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    let mut state = super::BatchRebalance {
        input: stream::empty().boxed(),
        batch_size: 8,
        pending: std::collections::VecDeque::new(),
        pending_rows: 0,
        ready: std::collections::VecDeque::new(),
        done: false,
    };
    let mut emitted = 0;
    for count in 1..=1000 {
        state.push(batch.clone())?;
        assert!(state.pending_rows < 18);
        assert!(state.pending.len() < 18);
        for batch in state.ready.drain(..) {
            assert_eq!(batch.num_rows() % 8, 0);
            emitted += batch.num_rows();
        }
        assert_eq!(emitted + state.pending_rows, count);
    }
    Ok(())
}

#[tokio::test]
async fn rebalancing_yields_before_a_second_full_tail_arrives() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(18)),
    )?;
    let input = stream::iter(vec![Ok(batch)])
        .chain(stream::pending())
        .boxed();
    let mut output = rebalance_batches(input, 8);
    let first = output
        .next()
        .now_or_never()
        .expect("first batch should not wait for another input")
        .expect("expected a batch")?;
    assert_eq!(first.num_rows(), 8);
    Ok(())
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(3)]
#[case(4)]
#[case(8)]
#[case(64)]
#[tokio::test]
async fn rebalanced_tails_exceed_the_coalescing_threshold(
    #[case] batch_size: usize,
) -> VortexResult<()> {
    let reserve = (batch_size / 2 + 1) * 2;
    for rows in reserve..batch_size * 5 {
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?;
        let batches: Vec<RecordBatch> =
            rebalance_batches(stream::iter(vec![Ok(batch)]).boxed(), batch_size)
                .try_collect()
                .await?;
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            rows
        );
        assert!(
            batches
                .iter()
                .all(|batch| batch.num_rows() > batch_size / 2)
        );
    }
    Ok(())
}
