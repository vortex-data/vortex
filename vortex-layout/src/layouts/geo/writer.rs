// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Strategy that writes data with an embedded geospatial index.
//!
//! This strategy expects to receive a chunk stream of `BINARY` data that corresponds to WKB
//! encoded geometry objects. Each chunk yields a new bloom filter composed of the H3 cell IDs for
//! all geometries in the chunk. This allows us to do very fast handling of contains queries,
//! without needing to read the full data or decode the WKBs until after a large pruning step.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use fastbloom::BloomFilter;
use futures::StreamExt;
use geo::ConvexHull;
use h3o::Resolution;
use h3o::geom::ContainmentMode;
use h3o::geom::TilerBuilder;
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
use vortex_error::vortex_panic;
use vortex_io::runtime::Handle;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::geo::GeoLayout;
use crate::layouts::geo::make_geom;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

#[derive(Clone)]
pub struct GeoStrategy {
    data_child: Arc<dyn LayoutStrategy>,
    rtree_child: Arc<dyn LayoutStrategy>,
    zone_len: usize,
}

impl GeoStrategy {
    pub fn new(data_child: Arc<dyn LayoutStrategy>, rtree_child: Arc<dyn LayoutStrategy>, zone_len: usize) -> Self {
        Self {
            data_child,
            rtree_child,
            zone_len,
        }
    }
}

#[async_trait]
impl LayoutStrategy for GeoStrategy {
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
        let accum = H3BloomAccumulator::new();

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

        Ok(GeoLayout::new(data_layout, rtree_layout, n_trees).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.data_child.buffered_bytes()
    }
}

#[derive(Clone)]
struct H3BloomAccumulator {
    inner: Arc<Mutex<VarBinBuilder<u32>>>,
}

impl H3BloomAccumulator {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VarBinBuilder::with_capacity(128))),
        }
    }
}

impl H3BloomAccumulator {
    fn push_chunk(&self, chunk: &ArrayRef) {
        let chunk = chunk.to_varbinview();
        chunk.with_iterator(|iter| {
            let mut tiler = TilerBuilder::new(Resolution::Eight)
                .containment_mode(ContainmentMode::Covers)
                .build();

            for geom in iter.filter_map(|v| make_geom(v?)) {
                // We use the H3 library to turn each geometry into a set of H3 cells that
                // fully cover the geometry.
                // TODO(aduffy): how expensive is this?
                let cv = geom.convex_hull();

                // Add to the tiler
                tiler
                    .add(cv)
                    .unwrap_or_else(|e| vortex_panic!("Failed to tile geometry: {e}"));
            }
            // TODO(aduffy): tweak these params
            let mut filter = BloomFilter::with_false_pos(0.01).expected_items(8192);
            for cell_id in tiler.into_coverage() {
                filter.insert_hash(cell_id.into());
            }

            // Return the serialized copy of the bloom filter
            let encoded = bincode::serde::encode_to_vec(&filter, bincode::config::standard())
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
