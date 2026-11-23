// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow output adapters for scan drivers.

use std::future;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_scan::execution::driver::ScanDriver;
use vortex_scan::prepare::ScanBuilder;

use crate::ArrowSessionExt;
use crate::ExecuteArrowExt;

/// Arrow record-batch output for a scan driver.
pub trait ArrowScanExt {
    /// Runs the scan, executing each batch to an Arrow [`RecordBatch`].
    ///
    /// When `schema` is `None`, the session derives it from the scan's output dtype. Conversion
    /// and decompression run on the worker producing each batch, in parallel across chunks.
    fn run_record_batches(
        self,
        builder: impl Into<ScanBuilder>,
        schema: Option<SchemaRef>,
    ) -> BoxStream<'static, VortexResult<RecordBatch>>;
}

impl ArrowScanExt for ScanDriver {
    fn run_record_batches(
        self,
        builder: impl Into<ScanBuilder>,
        schema: Option<SchemaRef>,
    ) -> BoxStream<'static, VortexResult<RecordBatch>> {
        let builder = builder.into();
        let session = builder.session().clone();
        let schema = match schema {
            Some(schema) => schema,
            None => match session.arrow().to_arrow_schema(builder.dtype()) {
                Ok(schema) => Arc::new(schema),
                Err(error) => return stream::once(future::ready(Err(error))).boxed(),
            },
        };
        let field = Field::new_struct("", schema.fields().clone(), false);
        self.run_mapped(builder, move |batch| {
            let mut ctx = session.create_execution_ctx();
            let arrow = batch.execute_arrow(Some(&field), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        })
    }
}
