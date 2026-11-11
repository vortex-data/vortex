// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Detect columns whose values are all the same and write them as one constant chunk.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A strategy that writes a column with one value in all rows as a single constant chunk.
///
/// Downstream strategies split a stream into blocks by row count and by uncompressed size. For a
/// constant column, each block compresses to the same constant, so these splits only add chunks,
/// segments and zone maps that have no use.
///
/// This strategy reads chunks while they all hold the same value. If the stream ends in this
/// state, it writes one [`ConstantArray`] with the `constant` strategy. If a chunk holds a
/// different value, it gives the stream to the `child` strategy. The rows read until then go first,
/// as one [`ConstantArray`].
///
/// While it reads, the strategy keeps only the sequence ID of the most recent chunk. It drops the
/// earlier IDs, so that other columns can continue to write their segments.
#[derive(Clone)]
pub struct ConstantStrategy {
    child: Arc<dyn LayoutStrategy>,
    constant: Arc<dyn LayoutStrategy>,
}

impl ConstantStrategy {
    /// Create a strategy that writes constant columns with `constant` and all other columns with
    /// `child`.
    pub fn new<C: LayoutStrategy, K: LayoutStrategy>(child: C, constant: K) -> Self {
        Self {
            child: Arc::new(child),
            constant: Arc::new(constant),
        }
    }
}

/// The rows read so far, which all hold the same value.
struct ConstantRun {
    /// The sequence ID of the most recent chunk in the run.
    sequence_id: SequenceId,
    scalar: Scalar,
    len: usize,
}

impl ConstantRun {
    fn into_item(self) -> (SequenceId, ArrayRef) {
        let array = ConstantArray::new(self.scalar, self.len).into_array();
        (self.sequence_id, array)
    }
}

#[async_trait]
impl LayoutStrategy for ConstantStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        if !supports_constant_detection(&dtype) {
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let mut exec_ctx = session.create_execution_ctx();
        let mut run: Option<ConstantRun> = None;
        let mut mismatch = None;

        while let Some(item) = stream.next().await {
            let (sequence_id, chunk) = item?;

            // An empty chunk adds no rows. Drop its sequence ID now.
            if chunk.is_empty() {
                continue;
            }

            if !is_constant(&chunk, &mut exec_ctx)? {
                mismatch = Some((sequence_id, chunk));
                break;
            }

            let scalar = match chunk.as_constant() {
                Some(scalar) => scalar.clone(),
                None => chunk.execute_scalar(0, &mut exec_ctx)?,
            };

            match run.as_mut() {
                None => {
                    run = Some(ConstantRun {
                        sequence_id,
                        scalar,
                        len: chunk.len(),
                    });
                }
                Some(current) if current.scalar == scalar => {
                    // Replacing the sequence ID drops the earlier one.
                    current.sequence_id = sequence_id;
                    current.len += chunk.len();
                }
                Some(_) => {
                    mismatch = Some((sequence_id, chunk));
                    break;
                }
            }
        }

        let Some((sequence_id, chunk)) = mismatch else {
            // The stream ended, and all of its rows hold the same value.
            let Some(run) = run else {
                return self
                    .child
                    .write_stream(ctx, segment_sink, stream, eof, session)
                    .await;
            };

            let constant_stream =
                SequentialStreamAdapter::new(dtype, stream::iter([Ok(run.into_item())]))
                    .sendable();

            return self
                .constant
                .write_stream(ctx, segment_sink, constant_stream, eof, session)
                .await;
        };

        // The column is not constant. The child gets the rows read so far, then the rest of the
        // stream.
        let prefix = run
            .map(ConstantRun::into_item)
            .into_iter()
            .chain([(sequence_id, chunk)])
            .map(Ok);
        let replayed =
            SequentialStreamAdapter::new(dtype, stream::iter(prefix).chain(stream)).sendable();

        self.child
            .write_stream(ctx, segment_sink, replayed, eof, session)
            .await
    }
}

/// Whether [`is_constant`] can examine arrays of `dtype` without an error.
fn supports_constant_detection(dtype: &DType) -> bool {
    !matches!(dtype, DType::Union(..) | DType::Variant(..))
}


#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::layouts::chunked::Chunked;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::Flat;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::repartition::RepartitionStrategy;
    use crate::layouts::repartition::RepartitionWriterOptions;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    const BLOCK_LEN: usize = 8192;

    /// Writes `chunks` with a [`ConstantStrategy`] in front of an 8K row repartition, reads the
    /// layout back, and checks that it holds the same rows.
    fn round_trip(chunks: Vec<ArrayRef>, dtype: DType) -> VortexResult<LayoutRef> {
        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();

        let child = RepartitionStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: BLOCK_LEN,
                block_size_target: None,
                canonicalize: false,
            },
        );
        let strategy = ConstantStrategy::new(child, FlatLayoutStrategy::default());

        block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();

            let layout = strategy
                .write_stream(
                    ArrayContext::empty().into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.clone().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
            let expr = root().bind(reader.dtype())?;
            let result = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(usize::try_from(layout.row_count())?),
                )?
                .await?;

            let mut exec_ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(result, array, &mut exec_ctx);

            Ok(layout)
        })
    }

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, NonNullable)
    }

    fn repeated(value: i64, len: usize) -> ArrayRef {
        PrimitiveArray::from_iter(iter::repeat_n(value, len)).into_array()
    }

    #[test]
    fn constant_chunks_become_one_flat_layout() -> VortexResult<()> {
        let chunks = (0..10).map(|_| repeated(7, BLOCK_LEN)).collect();

        let layout = round_trip(chunks, i64_dtype())?;

        assert!(layout.is::<Flat>());
        assert_eq!(layout.row_count(), 10 * BLOCK_LEN as u64);
        Ok(())
    }

    #[test]
    fn constant_array_is_not_split() -> VortexResult<()> {
        let chunks = vec![ConstantArray::new(42i64, 100 * BLOCK_LEN).into_array()];

        let layout = round_trip(chunks, i64_dtype())?;

        assert!(layout.is::<Flat>());
        Ok(())
    }

    #[test]
    fn all_null_column_becomes_one_flat_layout() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullable);
        let chunks = (0..3)
            .map(|_| ConstantArray::new(Scalar::null(dtype.clone()), BLOCK_LEN).into_array())
            .collect();

        let layout = round_trip(chunks, dtype)?;

        assert!(layout.is::<Flat>());
        Ok(())
    }

    #[test]
    fn empty_chunks_do_not_break_a_constant_run() -> VortexResult<()> {
        let chunks = vec![
            repeated(5, BLOCK_LEN),
            repeated(5, 0),
            ConstantArray::new(5i64, BLOCK_LEN).into_array(),
        ];

        let layout = round_trip(chunks, i64_dtype())?;

        assert!(layout.is::<Flat>());
        Ok(())
    }

    #[test]
    fn non_constant_chunk_replays_constant_prefix() -> VortexResult<()> {
        let chunks = vec![
            ConstantArray::new(1i64, BLOCK_LEN).into_array(),
            repeated(1, BLOCK_LEN),
            PrimitiveArray::from_iter((0..BLOCK_LEN as i64).map(|i| i % 3)).into_array(),
            ConstantArray::new(1i64, BLOCK_LEN).into_array(),
        ];

        let layout = round_trip(chunks, i64_dtype())?;

        assert!(layout.is::<Chunked>());
        assert_eq!(layout.nchildren(), 4);
        Ok(())
    }

    #[test]
    fn different_constant_values_use_child() -> VortexResult<()> {
        let chunks = vec![
            ConstantArray::new(1i64, BLOCK_LEN).into_array(),
            ConstantArray::new(2i64, BLOCK_LEN).into_array(),
        ];

        let layout = round_trip(chunks, i64_dtype())?;

        assert!(layout.is::<Chunked>());
        assert_eq!(layout.nchildren(), 2);
        Ok(())
    }

    #[test]
    fn empty_stream_uses_child() -> VortexResult<()> {
        let layout = round_trip(vec![], i64_dtype())?;

        assert_eq!(layout.row_count(), 0);
        Ok(())
    }
}
