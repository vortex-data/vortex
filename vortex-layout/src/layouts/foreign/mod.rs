// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Layout;
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
        _ctx: &ReadContext,
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

impl Layout for ForeignLayout {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_layout(&self) -> LayoutRef {
        Arc::new(self.clone())
    }

    fn encoding(&self) -> LayoutEncodingRef {
        self.encoding.clone()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn nchildren(&self) -> usize {
        self.children.len()
    }

    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.children.get(idx).cloned().ok_or_else(|| {
            vortex_err!("Child index out of bounds: {} of {}", idx, self.nchildren())
        })
    }

    fn child_type(&self, idx: usize) -> LayoutChildType {
        LayoutChildType::Auxiliary(format!("[{idx}]").into())
    }

    fn metadata(&self) -> Vec<u8> {
        self.metadata.clone()
    }

    fn segment_ids(&self) -> Vec<SegmentId> {
        self.segment_ids.clone()
    }

    fn new_reader(
        &self,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        vortex_bail!(
            "Cannot read unknown layout encoding '{}'",
            self.encoding.id()
        )
    }
}
