// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::StructFields;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowScanExt;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_expr::col;
use vortex_expr::root;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use super::ScanDriver;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

async fn file() -> VortexResult<VortexFile> {
    let chunks = 8;
    let numbers = ChunkedArray::from_iter(
        (0..chunks).map(|chunk| buffer![chunk, chunk + 1, chunk + 2, chunk + 3].into_array()),
    )
    .into_array();
    let strings = ChunkedArray::from_iter(
        (0..chunks).map(|_| VarBinArray::from(vec!["a", "bb", "ccc", "dddd"]).into_array()),
    )
    .into_array();
    let array = StructArray::from_fields(&[("numbers", numbers), ("strings", strings)])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf)
}

fn driver() -> ScanDriver {
    ScanDriver::new(TokioRuntime::current(), 2, true)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_batches_match_arrays() -> VortexResult<()> {
    let file = file().await?;
    let scan = || ScanBuilder::new(SESSION.clone(), file.scan_file());
    let schema = Arc::new(SESSION.arrow().to_arrow_schema(scan().dtype())?);
    let field = Field::new_struct("", schema.fields().clone(), false);

    let arrays: Vec<ArrayRef> = driver().run(scan()).try_collect().await?;
    let mut ctx = SESSION.create_execution_ctx();
    let expected = arrays
        .into_iter()
        .map(|array| {
            let arrow = array.execute_arrow(Some(&field), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let derived: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), None)
        .try_collect()
        .await?;
    let given: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), Some(schema))
        .try_collect()
        .await?;

    assert_eq!(derived, expected);
    assert_eq!(given, expected);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reads_fieldless_nullable_struct() -> VortexResult<()> {
    let n = 10;
    for validity in [
        Validity::AllValid,
        Validity::from_iter((0..n).map(|i| i % 3 != 0)),
    ] {
        let empty = StructArray::try_new_with_dtype(vec![], StructFields::empty(), n, validity)?
            .into_array();
        let id = PrimitiveArray::from_iter(0..n as u64).into_array();
        let table = StructArray::from_fields(&[("id", id), ("c", empty.clone())])?.into_array();

        let mut buf = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut buf, table.to_array_stream())
            .await?;
        let file = SESSION.open_options().open_buffer(buf)?;

        let run = |projection: Expression| {
            let file = file.clone();
            async move {
                let scan = ScanBuilder::new(SESSION.clone(), file.scan_file())
                    .with_projection(projection.bind(file.dtype())?.optimize_recursive()?);
                let dtype = scan.dtype().clone();
                ArrayStreamAdapter::new(dtype, driver().run(scan))
                    .read_all()
                    .await
            }
        };

        let mut ctx = SESSION.create_execution_ctx();
        let projected = run(col("c")).await?;
        assert_eq!(projected.len(), n);
        assert_arrays_eq!(projected, empty, &mut ctx);

        let full = run(root()).await?;
        assert_arrays_eq!(full, table, &mut ctx);
    }
    Ok(())
}
