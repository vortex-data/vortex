// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod reader;
pub mod writer;

use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::segments::SegmentSource;

/// Chunked layout vtable.
#[derive(Clone, Debug)]
pub struct Chunked;

/// Backwards-compatible name for the chunked layout plugin.
pub use Chunked as ChunkedLayoutEncoding;

/// Chunked-layout-specific data.
#[derive(Clone, Debug)]
pub struct ChunkedData {
    chunk_offsets: Vec<u64>,
}

/// A layout partitioned into independently readable row chunks.
pub type ChunkedLayout = Layout<Chunked>;

impl VTable for Chunked {
    type LayoutData = ChunkedData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.chunked");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let chunk_offsets = chunk_offsets(args.children)?;
        if chunk_offsets.last().copied() != Some(args.row_count) {
            vortex_bail!("Chunked child row counts do not add up to parent row count");
        }
        Ok(ChunkedData { chunk_offsets })
    }

    fn child_dtype(layout: &Layout<Self>, _idx: usize) -> VortexResult<DType> {
        Ok(layout.dtype().clone())
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        LayoutChildType::Chunk((idx, layout.chunk_offsets[idx]))
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ChunkedReader::new(
            layout.clone(),
            name,
            segment_source,
            session,
            ctx.clone(),
        )))
    }
}

/// Convenience methods for [`ChunkedLayout`].
pub trait ChunkedLayoutExt: Sized {
    /// Construct a chunked layout.
    fn new(row_count: u64, dtype: DType, children: Arc<dyn LayoutChildren>) -> Self;

    /// Rebuild this layout with owned children.
    fn with_children(&self, children: Vec<LayoutRef>) -> Self;
}

impl ChunkedLayoutExt for ChunkedLayout {
    fn new(row_count: u64, dtype: DType, children: Arc<dyn LayoutChildren>) -> Self {
        let offsets = chunk_offsets(children.as_ref()).vortex_expect("chunk row counts overflow");
        assert_eq!(
            offsets.last().copied(),
            Some(row_count),
            "Row count mismatch"
        );
        LayoutParts::new(
            Chunked,
            dtype,
            row_count,
            Vec::new(),
            children,
            ChunkedData {
                chunk_offsets: offsets,
            },
        )
        .into_typed()
    }

    fn with_children(&self, children: Vec<LayoutRef>) -> Self {
        Self::new(
            self.row_count(),
            self.dtype().clone(),
            OwnedLayoutChildren::layout_children(children),
        )
    }
}

fn chunk_offsets(children: &dyn LayoutChildren) -> VortexResult<Vec<u64>> {
    let mut offsets = Vec::with_capacity(children.nchildren() + 1);
    offsets.push(0u64);
    for idx in 0..children.nchildren() {
        offsets.push(
            offsets[idx]
                .checked_add(children.child_row_count(idx))
                .ok_or_else(|| vortex_err!("Chunked child row counts overflow"))?,
        );
    }
    Ok(offsets)
}
