// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::DynLayout;
use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutEncoding;
use crate::LayoutEncodingId;
use crate::LayoutEncodingRef;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Placeholder layout encoding used when deserializing an unknown layout encoding ID.
#[derive(Clone, Debug)]
pub(crate) struct ForeignLayoutEncoding {
    id: LayoutEncodingId,
}

impl ForeignLayoutEncoding {
    pub(crate) fn new(id: LayoutEncodingId) -> Self {
        Self { id }
    }
}

impl LayoutEncoding for ForeignLayoutEncoding {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutEncodingId {
        self.id
    }

    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef> {
        let child_layouts = (0..children.nchildren())
            .map(|idx| children.child(idx, dtype))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(new_foreign_layout(
            self.id,
            dtype.clone(),
            row_count,
            metadata.to_vec(),
            segment_ids,
            child_layouts,
        ))
    }
}

/// Placeholder layout used when deserializing an unknown layout encoding ID.
#[derive(Clone, Debug)]
pub(crate) struct ForeignLayout {
    encoding: LayoutEncodingRef,
    dtype: DType,
    row_count: u64,
    metadata: Vec<u8>,
    segment_ids: Vec<SegmentId>,
    children: Vec<LayoutRef>,
}

impl ForeignLayout {
    pub(crate) fn new(
        encoding_id: LayoutEncodingId,
        dtype: DType,
        row_count: u64,
        metadata: Vec<u8>,
        segment_ids: Vec<SegmentId>,
        children: Vec<LayoutRef>,
    ) -> Self {
        let encoding =
            LayoutEncodingRef::new_arc(Arc::new(ForeignLayoutEncoding::new(encoding_id)));

        Self {
            encoding,
            dtype,
            row_count,
            metadata,
            segment_ids,
            children,
        }
    }
}

pub(crate) fn new_foreign_layout(
    encoding_id: LayoutEncodingId,
    dtype: DType,
    row_count: u64,
    metadata: Vec<u8>,
    segment_ids: Vec<SegmentId>,
    children: Vec<LayoutRef>,
) -> LayoutRef {
    Arc::new(ForeignLayout::new(
        encoding_id,
        dtype,
        row_count,
        metadata,
        segment_ids,
        children,
    ))
}

impl DynLayout for ForeignLayout {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_to_layout(&self) -> LayoutRef {
        Arc::new(self.clone())
    }

    fn dyn_encoding_id(&self) -> LayoutEncodingId {
        self.encoding.id()
    }

    fn dyn_row_count(&self) -> u64 {
        self.row_count
    }

    fn dyn_dtype(&self) -> &DType {
        &self.dtype
    }

    fn dyn_nchildren(&self) -> usize {
        self.children.len()
    }

    fn dyn_nslots(&self) -> usize {
        // A foreign layout is opaque: its children are dense, so each slot is always present.
        self.children.len()
    }

    fn dyn_slot(&self, slot: usize) -> VortexResult<Option<LayoutRef>> {
        Ok(self.children.get(slot).cloned())
    }

    fn dyn_slot_type(&self, slot: usize) -> Option<LayoutChildType> {
        (slot < self.children.len())
            .then(|| LayoutChildType::Auxiliary(format!("[{slot}]").into()))
    }

    fn dyn_metadata(&self) -> Vec<u8> {
        self.metadata.clone()
    }

    fn dyn_segment_ids(&self) -> Vec<SegmentId> {
        self.segment_ids.clone()
    }

    fn dyn_new_reader(
        &self,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
        _ctx: &crate::LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        vortex_bail!(
            "Cannot read unknown layout encoding '{}'",
            self.encoding.id()
        )
    }
}
