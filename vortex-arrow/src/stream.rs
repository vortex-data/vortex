// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow output for array streams.

use std::future;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::VortexSessionExecute;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::ArrowSessionExt;
use crate::ExecuteArrowExt;

/// Arrow record-batch output for an [`ArrayStream`], such as a piece of a split scan.
pub trait ArrowArrayStreamExt: ArrayStream + Send + Sized + 'static {
    /// Converts each array of the stream to an Arrow [`RecordBatch`] with `schema`.
    ///
    /// Each conversion runs as a CPU task on the session's runtime, with at most `concurrency`
    /// in flight, and the batches keep the stream's order. When `schema` is `None`, the session
    /// derives it from the stream's dtype.
    fn to_arrow(
        self,
        session: &VortexSession,
        schema: Option<SchemaRef>,
        concurrency: NonZeroUsize,
    ) -> BoxStream<'static, VortexResult<RecordBatch>> {
        let schema = match schema {
            Some(schema) => schema,
            None => match session.arrow().to_arrow_schema(self.dtype()) {
                Ok(schema) => Arc::new(schema),
                Err(error) => return stream::once(future::ready(Err(error))).boxed(),
            },
        };
        let field = Arc::new(Field::new_struct("", schema.fields().clone(), false));
        let session = session.clone();
        let handle = session.handle();
        self.map(move |array| {
            let session = session.clone();
            let field = Arc::clone(&field);
            handle.spawn_cpu(move || {
                let arrow =
                    array?.execute_arrow(Some(&field), &mut session.create_execution_ctx())?;
                Ok(RecordBatch::from(arrow.as_struct().clone()))
            })
        })
        .buffered(concurrency.get())
        .boxed()
    }
}

impl<S: ArrayStream + Send + 'static> ArrowArrayStreamExt for S {}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use futures::TryStreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    use super::ArrowArrayStreamExt;

    #[test]
    fn converts_each_array_in_order() -> VortexResult<()> {
        let arrays = [buffer![0i32, 1], buffer![2i32], buffer![3i32, 4, 5]]
            .into_iter()
            .map(|values| {
                StructArray::from_fields(&[("a", values.into_array())]).map(IntoArray::into_array)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let dtype = arrays[0].dtype().clone();
        let batches: Vec<_> = block_on(|handle| {
            let session = VortexSession::empty().with_handle(handle);
            ArrayStreamAdapter::new(dtype, stream::iter(arrays.into_iter().map(Ok)))
                .to_arrow(
                    &session,
                    None,
                    NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
                )
                .try_collect()
        })?;
        let values: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(values, [0, 1, 2, 3, 4, 5]);
        Ok(())
    }
}
