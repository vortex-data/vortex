//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
mod writer;

use std::sync::Arc;

pub use reader::*;
use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_expr::{Identifier, ScopeDType};

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef,
    VTable, vtable,
};

vtable!(View);

impl VTable for ViewVTable {
    type Layout = ViewLayout;
    type Encoding = ViewLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        VIEW_LAYOUT_ID.clone()
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ViewLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout
            .scope_dtype
            .dtype(&Identifier::Identity)
            .vortex_expect("view layout always has an identity")
    }

    fn scope_dtype(layout: &Self::Layout) -> &ScopeDType {
        &layout.scope_dtype
    }

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        let mut segments = Vec::with_capacity(1 + layout.buffers.len());
        segments.push(layout.views);
        segments.extend(layout.buffers.iter());

        segments
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn child(_layout: &Self::Layout, _idx: usize) -> VortexResult<LayoutRef> {
        vortex_bail!("ViewLayout has no children")
    }

    fn child_type(_layout: &Self::Layout, _idx: usize) -> LayoutChildType {
        vortex_panic!("ViewLayout has no children")
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ViewReader::new(
            layout.clone(),
            name,
            segment_source,
            ctx,
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        mut segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.is_empty() {
            vortex_bail!("ViewLayout must have at least one segment to hold views");
        }

        let views = segment_ids.remove(0);
        let scope_dtype = ScopeDType::new(dtype.clone());
        Ok(ViewLayout::new(row_count, scope_dtype, views, segment_ids))
    }
}

pub static VIEW_LAYOUT_ID: LayoutId = LayoutId::new_ref("vortex.view");

#[derive(Debug)]
pub struct ViewLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ViewLayout {
    row_count: u64,
    scope_dtype: ScopeDType,
    views: SegmentId,
    buffers: Vec<SegmentId>,
}

impl ViewLayout {
    pub fn new(
        row_count: u64,
        scope_dtype: ScopeDType,
        views: SegmentId,
        buffers: Vec<SegmentId>,
    ) -> Self {
        Self {
            row_count,
            scope_dtype,
            views,
            buffers,
        }
    }
}
