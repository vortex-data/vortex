// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion::arrow::array::RecordBatch;
use datafusion_common::ScalarValue;

use crate::common_tests::TestSessionContext;

// https://github.com/vortex-data/vortex/issues/6211
#[tokio::test]
async fn test_cast_int_to_string() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    ctx.session
        .sql(r#"copy (select 1 as id) to 'example.vortex'"#)
        .await?
        .show()
        .await?;

    let batches = ctx
        .session
        .sql(r#"select cast(id as string) as sid from 'example.vortex' where id > 0"#)
        .await?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let actual = ScalarValue::try_from_array(batches[0].column(0).as_ref(), 0)?;
    assert!(
        matches!(&actual, ScalarValue::Utf8(Some(value)) | ScalarValue::Utf8View(Some(value)) if value == "1")
    );

    let batches = ctx
        .session
        .sql(r#"select id from 'example.vortex' where cast (id as string) == '1'"#)
        .await?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let actual = ScalarValue::try_from_array(batches[0].column(0).as_ref(), 0)?;
    assert_eq!(actual, ScalarValue::Int64(Some(1)));

    // This fails as it pushes string cast to the scan
    let batches = ctx
        .session
        .sql(r#"select cast(id as string) from 'example.vortex'"#)
        .await?
        .collect()
        .await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let actual = ScalarValue::try_from_array(batches[0].column(0).as_ref(), 0)?;
    assert!(
        matches!(&actual, ScalarValue::Utf8(Some(value)) | ScalarValue::Utf8View(Some(value)) if value == "1")
    );

    Ok(())
}
