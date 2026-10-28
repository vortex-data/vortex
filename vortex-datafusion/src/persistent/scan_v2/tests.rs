// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use super::align_batches;

#[rstest]
#[case(0, vec![3, 0, 10, 2])]
#[case(1, vec![3, 10, 2])]
#[case(4, vec![4, 8, 3])]
#[case(32, vec![15])]
#[tokio::test]
async fn alignment_preserves_rows_and_nulls(
    #[case] batch_size: usize,
    #[case] expected_lengths: Vec<usize>,
) -> VortexResult<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, true)]));
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
    let batches: Vec<RecordBatch> = align_batches(stream::iter(input).boxed(), batch_size)
        .try_collect()
        .await?;
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
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
async fn alignment_preserves_zero_column_row_counts() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )?;
    let input = stream::iter(vec![Ok(batch.clone()), Ok(batch)]).boxed();
    let batches: Vec<RecordBatch> = align_batches(input, 4).try_collect().await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(), vec![4, 2]);
    assert!(batches.iter().all(|batch| batch.num_columns() == 0));
    Ok(())
}

#[tokio::test]
async fn alignment_propagates_errors_with_a_pending_tail() -> VortexResult<()> {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )?;
    let input = stream::iter(vec![Ok(batch), Err(vortex_err!("input failed"))]).boxed();
    let mut aligned = align_batches(input, 4);
    let error = aligned.next().await.expect("expected input error").expect_err("input failed");
    assert!(error.to_string().contains("input failed"));
    assert!(aligned.next().await.is_none());
    Ok(())
}
