// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::request::ScanRequest;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutDeserializeArgs;
use crate::layout_v2::LayoutScanPlanCtx;
use crate::layout_v2::VTable;
use crate::layout_v2::metadata_bytes_field;
use crate::scan::v2::layouts::flat as scan_flat;
use crate::segments::SegmentId;

/// V2 flat layout vtable.
#[derive(Clone, Debug)]
pub struct Flat;

/// V2 flat layout data.
#[derive(Clone, Debug)]
pub struct FlatData {
    pub(crate) segment_id: SegmentId,
    pub(crate) array_ctx: ReadContext,
    pub(crate) array_tree: Option<ByteBuffer>,
}

impl FlatData {
    /// Returns the serialized array segment ID.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the array read context.
    pub fn array_ctx(&self) -> &ReadContext {
        &self.array_ctx
    }

    /// Returns the optional inline array encoding tree.
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.array_tree.as_ref()
    }
}

impl VTable for Flat {
    type LayoutData = FlatData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.flat")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.segment_ids.len() == 1,
            "Flat layout must have exactly one segment ID"
        );
        Ok(FlatData {
            segment_id: args.segment_ids[0],
            array_ctx: args.array_ctx.clone(),
            array_tree: metadata_bytes_field(args.metadata, 1)?.map(ByteBuffer::from),
        })
    }

    fn child_dtype(_layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        vortex_bail!("Flat layout has no child {idx}")
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        vortex_bail!("Flat layout has no child {idx}")
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        ctx: &LayoutScanPlanCtx,
    ) -> VortexResult<ScanPlanRef> {
        scan_flat::new_scan_plan(layout, req, ctx)
    }
}
