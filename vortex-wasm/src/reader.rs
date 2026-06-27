// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`WasmReader`] drives an embedded kernel to decode a [`WasmLayout`].
//!
//! Child layouts are decoded eagerly through the normal layout reader machinery, encoded into
//! [`CanonicalMessage`](crate::message)s, and served to the kernel through the `vx_decode_child`
//! host import. The kernel's canonical output is then sliced/filtered/projected like any other
//! reader.
//!
//! WASM layouts are **decode-only**: the kernel decompresses, nothing more. There is deliberately
//! no pushdown — filters and projections are evaluated on the fully decoded array (exactly as a
//! [`FlatLayout`](vortex_layout::layouts::flat) leaf does), never pushed into the kernel, and there
//! is no statistics-based pruning. This keeps kernels simple and untrusted file code off the
//! query-planning path.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::RowSplits;
use vortex_layout::SplitRange;
use vortex_layout::segments::SegmentSource;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::HostDecoder;
use crate::WasmKernel;
use crate::layout::WasmLayout;
use crate::message::encode_canonical;

/// Stateful reader for a [`WasmLayout`].
pub struct WasmReader {
    layout: WasmLayout,
    name: Arc<str>,
    session: VortexSession,
    segment_source: Arc<dyn SegmentSource>,
    /// One reader per child layout, in index order.
    children: Vec<LayoutReaderRef>,
}

impl WasmReader {
    /// Construct a reader, building child readers up front (propagating `ctx`).
    pub fn try_new(
        layout: WasmLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> VortexResult<Self> {
        let mut children = Vec::with_capacity(layout.child_layouts().len());
        for (idx, child_layout) in layout.child_layouts().iter().enumerate() {
            // Each child carries its own dtype, so the reader does not need the parent's.
            children.push(child_layout.new_reader(
                format!("{name}.input[{idx}]").into(),
                Arc::clone(&segment_source),
                &session,
                &ctx,
            )?);
        }
        Ok(Self {
            layout,
            name,
            session,
            segment_source,
            children,
        })
    }

    /// Build the full decoded output array by running the kernel.
    fn decoded_array_future(&self) -> ArrayFuture {
        let segment_source = Arc::clone(&self.segment_source);
        let kernel_segment = self.layout.kernel_segment();
        let payload_segment = self.layout.payload_segment();
        let session = self.session.clone();
        let children = self.children.clone();

        async move {
            let kernel_handle = segment_source.request(kernel_segment).await?;
            let kernel = WasmKernel::new(kernel_handle.to_host_sync().as_ref())?;

            // Eagerly decode each child input into a CanonicalMessage.
            let mut ctx = session.create_execution_ctx();
            let mut messages = Vec::with_capacity(children.len());
            for child in &children {
                let row_count = child.row_count();
                let len = usize::try_from(row_count)?;
                let array = child
                    .projection_evaluation(&(0..row_count), &root(), MaskFuture::new_true(len))?
                    .await?;
                let canonical = array.execute::<Canonical>(&mut ctx)?;
                messages.push(encode_canonical(&canonical, &mut ctx)?);
            }

            let payload = match payload_segment {
                Some(segment) => segment_source
                    .request(segment)
                    .await?
                    .to_host_sync()
                    .as_ref()
                    .to_vec(),
                None => Vec::new(),
            };

            let decoder = PrecomputedDecoder { messages };
            kernel.decode(&payload, &decoder)
        }
        .boxed()
    }
}

/// A [`HostDecoder`] backed by child arrays decoded up front.
struct PrecomputedDecoder {
    messages: Vec<Vec<u8>>,
}

impl HostDecoder for PrecomputedDecoder {
    fn decode_child(&self, node_index: usize) -> VortexResult<Vec<u8>> {
        self.messages.get(node_index).cloned().ok_or_else(|| {
            vortex_err!(
                "kernel requested child {node_index} but only {} are available",
                self.messages.len()
            )
        })
    }
}

impl LayoutReader for WasmReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype_ref()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count_val()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count_val())?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Decode-only: no statistics-based pruning. Return the mask unchanged.
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // Decode-only: fully decode, then evaluate the predicate on the decoded array. Nothing is
        // pushed into the kernel.
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let array = self.decoded_array_future();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let mut array = array.await?;
            let mask = mask.await?;
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }
            let array = array.apply(&expr)?;
            let mut ctx = session.create_execution_ctx();
            let array_mask = array.null_as_false().execute(&mut ctx)?;
            Ok(mask.bitand(&array_mask))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let array = self.decoded_array_future();
        let expr = expr.clone();

        Ok(async move {
            let mut array = array.await?;
            let mask = mask.await?;
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }
            if !mask.all_true() {
                array = array.filter(mask)?;
            }
            array.apply(&expr)
        }
        .boxed())
    }
}
