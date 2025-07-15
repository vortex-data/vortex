//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
mod writer;

use std::sync::Arc;

use arcref::ArcRef;
pub use reader::*;
use vortex_array::{ArrayContext, DeserializeMetadata, RawMetadata};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef,
    VTable, vtable,
};

vtable!(View);

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ValidityTag {
    NonNullable,
    AllValid,
    AllInvalid,
    Array,
}

impl VTable for ViewVTable {
    type Layout = ViewLayout;
    type Encoding = ViewLayoutEncoding;
    type Metadata = RawMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        ArcRef::new_ref("vortex.view")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ViewLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        RawMetadata(vec![layout.validity_tag as u8])
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        let mut segments = Vec::with_capacity(1 + layout.buffers.len());
        segments.push(layout.views);
        segments.extend(layout.buffers.iter());

        segments
    }

    fn nchildren(_: &Self::Layout) -> usize {
        0
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        if idx == 0 {
            if layout.children.nchildren() == 1 {
                layout
                    .children
                    .child(0, &DType::Bool(Nullability::NonNullable))
            } else {
                vortex_bail!(
                    "ViewLayout: cannot access validity child, layout has child count {}",
                    layout.children.nchildren()
                );
            }
        } else {
            vortex_bail!("ViewLayout: invalid child ordinal {idx}")
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        if idx == 0 {
            LayoutChildType::Auxiliary("validity".into())
        } else {
            vortex_panic!("Invalid child idx {idx} for ViewLayout");
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<LayoutReaderRef> {
        // Reader will internally cache whatever it needs for accessing a particular layout.
        // The LayoutReader is kept alive for how long?
        Ok(Arc::new(ViewReader::new(
            layout.clone(),
            name,
            segment_source,
            layout.ctx.clone(),
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        mut segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        let validity_tag = match metadata[0] {
            0 => ValidityTag::NonNullable,
            1 => ValidityTag::AllValid,
            2 => ValidityTag::AllInvalid,
            3 => ValidityTag::Array,
            invalid => vortex_bail!("Invalid value for ValidityTag {invalid}"),
        };

        if segment_ids.is_empty() {
            vortex_bail!("ViewLayout must have at least one segment to hold views");
        }

        let views = segment_ids.remove(0);

        Ok(ViewLayout::new(
            row_count,
            dtype.clone(),
            validity_tag,
            views,
            segment_ids,
            children.to_arc(),
            ctx,
        ))
    }
}

#[derive(Debug)]
pub struct ViewLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ViewLayout {
    row_count: u64,
    dtype: DType,
    validity_tag: ValidityTag,
    views: SegmentId,
    buffers: Vec<SegmentId>,
    // Handle to lookup children. This will contain
    // either a 0-th child (for validity) or nothing.
    children: Arc<dyn LayoutChildren>,
    ctx: ArrayContext,
}

impl ViewLayout {
    pub fn new(
        row_count: u64,
        dtype: DType,
        validity_tag: ValidityTag,
        views: SegmentId,
        buffers: Vec<SegmentId>,
        children: Arc<dyn LayoutChildren>,
        ctx: ArrayContext,
    ) -> Self {
        Self {
            row_count,
            dtype,
            validity_tag,
            views,
            buffers,
            children,
            ctx,
        }
    }
}
