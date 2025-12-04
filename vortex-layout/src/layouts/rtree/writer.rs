// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::StreamExt;
use itertools::Itertools;
use rstar::RTree;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::builder::VarBinBuilder;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::rtree::RTreeLayout;
use crate::layouts::rtree::make_geom;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

#[derive(Clone)]
pub struct RTreeStrategy {
    data_child: Arc<dyn LayoutStrategy>,
    rtree_child: Arc<dyn LayoutStrategy>,
}

impl RTreeStrategy {
    pub fn new(data_child: Arc<dyn LayoutStrategy>, rtree_child: Arc<dyn LayoutStrategy>) -> Self {
        Self {
            data_child,
            rtree_child,
        }
    }
}

#[async_trait]
impl LayoutStrategy for RTreeStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        // split off data stream write to complete first.
        let data_eof = eof.split_off();
        let accum = RTreeAccumulator::new();

        let accum2 = accum.clone();
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map(move |chunk| {
                // simple stream transformation that passes the chunk through to the accumulator
                if let Ok((_, ref geoms)) = chunk {
                    accum2.push_chunk(geoms);
                }
                chunk
            }),
        )
        .sendable();

        let data_layout = self
            .data_child
            .write_stream(
                ctx.clone(),
                segment_sink.clone(),
                stream,
                data_eof,
                handle.clone(),
            )
            .await?;

        // After child write completes, get back the inner type.
        let rtree = accum.finish();

        let n_trees = rtree.len();

        // Write the rtree to a separate node
        let rtree_stream = rtree
            .to_array_stream()
            .sequenced(eof.split_off())
            .sendable();

        let rtree_layout = self
            .rtree_child
            .write_stream(ctx, segment_sink, rtree_stream, eof, handle)
            .await?;

        Ok(RTreeLayout::new(data_layout, rtree_layout, n_trees).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.data_child.buffered_bytes()
    }
}

#[derive(Clone)]
struct RTreeAccumulator {
    inner: Arc<Mutex<VarBinBuilder<u32>>>,
}

impl RTreeAccumulator {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VarBinBuilder::with_capacity(128))),
        }
    }
}

impl RTreeAccumulator {
    fn push_chunk(&self, chunk: &ArrayRef) {
        let chunk = chunk.to_varbinview();
        chunk.with_iterator(|iter| {
            let geoms = iter.filter_map(|v| make_geom(v?)).collect_vec();
            let rtree = RTree::bulk_load(geoms);
            // Serialize the tree as a new value and push it into the builder.
            let encoded = bincode::serde::encode_to_vec(&rtree, bincode::config::standard())
                .expect("rtree serde");

            self.inner.lock().expect("poisoned").append_value(encoded);
        });
    }

    /// Finish the accumulator yielding the RTree table
    fn finish(&self) -> ArrayRef {
        // Slice and contain only these chunks.
        let mut inner = self.inner.lock().expect("poisoned");
        let builder = std::mem::take(&mut *inner);
        let column = builder.finish(DType::Binary(Nullability::Nullable));
        let len = column.len();

        // Build the output array
        StructArray::new(
            FieldNames::from(vec!["rtree"]),
            vec![column.into_array()],
            len,
            Validity::NonNullable,
        )
        .into_array()
    }
}
