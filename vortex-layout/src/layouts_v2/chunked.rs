// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutDeserializeArgs;
use crate::layout_v2::LayoutScanPlanCtx;
use crate::layout_v2::VTable;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::request::ScanRequest;
use crate::scan::v2::layouts::chunked as scan_chunked;

/// V2 chunked layout vtable.
#[derive(Clone, Debug)]
pub struct Chunked;

/// V2 chunked layout data.
#[derive(Clone, Debug)]
pub struct ChunkedData {
    pub(crate) chunk_offsets: Vec<u64>,
}

impl ChunkedData {
    /// Returns the cumulative chunk offsets.
    pub fn chunk_offsets(&self) -> &[u64] {
        &self.chunk_offsets
    }
}

impl VTable for Chunked {
    type LayoutData = ChunkedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.chunked")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        EmptyMetadata::deserialize(args.metadata)?;
        let mut chunk_offsets: Vec<u64> = Vec::with_capacity(args.children.nchildren() + 1);
        chunk_offsets.push(0);
        for idx in 0..args.children.nchildren() {
            let next = chunk_offsets[idx]
                .checked_add(args.children.child_row_count(idx)?)
                .ok_or_else(|| vortex_err!("Chunked child row counts overflow"))?;
            chunk_offsets.push(next);
        }
        vortex_ensure!(
            chunk_offsets.last().copied() == Some(args.row_count),
            "Chunked child row counts do not add up to parent row count"
        );
        Ok(ChunkedData { chunk_offsets })
    }

    fn child_dtype(layout: Layout<Self>, _idx: usize) -> VortexResult<DType> {
        Ok(layout.dtype().clone())
    }

    fn child_type(layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        if idx >= layout.nchildren() {
            vortex_bail!("Chunked child index out of bounds: {idx}");
        }
        let offset = *layout
            .data()
            .chunk_offsets
            .get(idx)
            .ok_or_else(|| vortex_err!("Chunked child index out of bounds: {idx}"))?;
        Ok(LayoutChildType::Chunk((idx, offset)))
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        ctx: &LayoutScanPlanCtx,
    ) -> VortexResult<ScanPlanRef> {
        scan_chunked::new_scan_plan(layout, req, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::layout_v2::LayoutChildren;
    use crate::layout_v2::LayoutParts;
    use crate::layout_v2::LayoutRef;

    #[derive(Debug)]
    struct TestChildren {
        row_counts: Vec<u64>,
    }

    impl LayoutChildren for TestChildren {
        fn child(&self, idx: usize, _dtype: &DType) -> VortexResult<LayoutRef> {
            vortex_bail!("test child {idx} is not materialized")
        }

        fn child_row_count(&self, idx: usize) -> VortexResult<u64> {
            self.row_counts
                .get(idx)
                .copied()
                .ok_or_else(|| vortex_err!("test child index out of bounds: {idx}"))
        }

        fn nchildren(&self) -> usize {
            self.row_counts.len()
        }
    }

    fn primitive_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn read_context() -> ReadContext {
        ReadContext::new([])
    }

    #[test]
    fn chunked_deserialize_rejects_row_count_overflow() {
        let dtype = primitive_dtype();
        let read_context = read_context();
        let session = VortexSession::empty();
        let args = LayoutDeserializeArgs {
            dtype: &dtype,
            row_count: 0,
            metadata: &[],
            segment_ids: Vec::new(),
            children: Arc::new(TestChildren {
                row_counts: vec![u64::MAX, 1],
            }),
            array_ctx: &read_context,
            session: &session,
        };

        assert!(VTable::deserialize(&Chunked, &args).is_err());
    }

    #[test]
    fn chunked_child_type_rejects_terminal_offset_index() {
        let dtype = primitive_dtype();
        let layout = LayoutParts::new(
            Chunked,
            dtype,
            1,
            Vec::new(),
            Arc::new(TestChildren {
                row_counts: vec![1],
            }),
            ChunkedData {
                chunk_offsets: vec![0, 1],
            },
        )
        .into_typed();

        assert!(layout.child_type(1).is_err());
    }
}
