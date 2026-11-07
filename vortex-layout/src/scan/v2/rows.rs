// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The root rows each segment of a file serves.

use std::ops::Range;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::LayoutRef;
use crate::plan::physical::pipeline::SegmentRows;
use crate::segments::SegmentId;

/// Indexes the stored layout once: every segment maps to the root rows whose readers can need
/// it, independently of any scan's selection or plans.
///
/// A child placed at a row offset of its parent serves its own rows of the parent's. Any other
/// child, such as a dictionary's values or a zone map, serves every row of its parent: its own
/// row numbers cannot be compared with root rows. A segment several layouts share serves the
/// rows of all of them.
pub(super) fn segment_rows(layout: &LayoutRef) -> VortexResult<SegmentRows> {
    let mut rows = FxHashMap::default();
    visit(layout, 0..layout.row_count(), true, &mut rows)?;
    Ok(Arc::new(rows))
}

fn visit(
    layout: &LayoutRef,
    rows: Range<u64>,
    mapped: bool,
    segments: &mut FxHashMap<SegmentId, Range<u64>>,
) -> VortexResult<()> {
    for segment in layout.segment_ids() {
        segments
            .entry(segment)
            .and_modify(|current| {
                current.start = current.start.min(rows.start);
                current.end = current.end.max(rows.end);
            })
            .or_insert_with(|| rows.clone());
    }
    for slot in 0..layout.nslots() {
        let Some(child) = layout.slot(slot)? else {
            continue;
        };
        let offset = layout.slot_type(slot).and_then(|kind| kind.row_offset());
        let (child_rows, child_mapped) = match (mapped, offset) {
            (true, Some(offset)) => {
                let start = rows
                    .start
                    .checked_add(offset)
                    .ok_or_else(|| vortex_err!("layout row offset overflow"))?;
                let end = start
                    .checked_add(child.row_count())
                    .ok_or_else(|| vortex_err!("layout row end overflow"))?;
                (start..end, true)
            }
            _ => (rows.clone(), false),
        };
        visit(&child, child_rows, child_mapped, segments)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::PType;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::layouts::dict::DictLayout;
    use crate::layouts::flat::FlatLayout;

    fn flat(id: u32, rows: u64) -> LayoutRef {
        FlatLayout::new(
            rows,
            DType::Primitive(PType::U32, false.into()),
            id.into(),
            ReadContext::new([]),
        )
        .into_layout()
    }

    fn chunks(children: Vec<LayoutRef>) -> LayoutRef {
        ChunkedLayout::new(
            children.iter().map(|child| child.row_count()).sum(),
            children[0].dtype().clone(),
            OwnedLayoutChildren::layout_children(children),
        )
        .into_layout()
    }

    #[test]
    fn nested_chunks_serve_root_rows() -> VortexResult<()> {
        let layout = chunks(vec![flat(0, 10), chunks(vec![flat(1, 20), flat(2, 30)])]);
        let rows = segment_rows(&layout)?;
        assert_eq!(rows[&SegmentId::from(0)], 0..10);
        assert_eq!(rows[&SegmentId::from(1)], 10..30);
        assert_eq!(rows[&SegmentId::from(2)], 30..60);
        Ok(())
    }

    #[test]
    fn dictionary_values_serve_every_row_of_their_dictionary() -> VortexResult<()> {
        let values = chunks(vec![flat(1, 2), flat(2, 3)]);
        let codes = chunks(vec![flat(3, 10), flat(4, 10)]);
        let dictionary = DictLayout::new(values, codes).into_layout();
        let layout = chunks(vec![flat(0, 10), dictionary, flat(5, 30)]);
        let rows = segment_rows(&layout)?;
        assert_eq!(rows[&SegmentId::from(1)], 10..30);
        assert_eq!(rows[&SegmentId::from(2)], 10..30);
        assert_eq!(rows[&SegmentId::from(3)], 10..20);
        assert_eq!(rows[&SegmentId::from(4)], 20..30);
        assert_eq!(rows[&SegmentId::from(5)], 30..60);
        Ok(())
    }

    #[test]
    fn a_shared_segment_serves_every_reader() -> VortexResult<()> {
        let layout = chunks(vec![flat(0, 20), flat(1, 20), flat(0, 20)]);
        let rows = segment_rows(&layout)?;
        assert_eq!(rows[&SegmentId::from(0)], 0..60);
        assert_eq!(rows[&SegmentId::from(1)], 20..40);
        Ok(())
    }
}
