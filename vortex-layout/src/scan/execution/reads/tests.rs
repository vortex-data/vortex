// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_buffer::Alignment;
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

fn locations(count: u64) -> Vec<SegmentLocation> {
    (0..count)
        .map(|id| SegmentLocation {
            offset: id * 4,
            length: 4,
            alignment: Alignment::none(),
        })
        .collect()
}

#[test]
fn nested_chunks_use_root_file_rows() -> VortexResult<()> {
    let layout = chunks(vec![flat(0, 10), chunks(vec![flat(1, 20), flat(2, 30)])]);
    let lifetimes = read_lifetimes(&layout, &locations(3))?;
    for (offset, end) in [(0, 10), (4, 30), (8, 60)] {
        assert_eq!(
            lifetimes[&IoTarget::range(offset, 4)],
            ReadLifetime::ThroughRow(end)
        );
    }
    Ok(())
}

#[test]
fn dictionary_values_inherit_the_entire_consumer_region() -> VortexResult<()> {
    let values = chunks(vec![flat(1, 2), flat(2, 3)]);
    let codes = chunks(vec![flat(3, 10), flat(4, 10)]);
    let dictionary = DictLayout::new(values, codes).into_layout();
    let layout = chunks(vec![flat(0, 10), dictionary, flat(5, 30)]);
    let lifetimes = read_lifetimes(&layout, &locations(6))?;
    // Both values chunks can be read by any row in the dictionary's root-row region 10..30.
    for offset in [4, 8] {
        assert_eq!(
            lifetimes[&IoTarget::range(offset, 4)],
            ReadLifetime::ThroughRow(30)
        );
    }
    assert_eq!(
        lifetimes[&IoTarget::range(12, 4)],
        ReadLifetime::ThroughRow(20)
    );
    assert_eq!(
        lifetimes[&IoTarget::range(16, 4)],
        ReadLifetime::ThroughRow(30)
    );
    Ok(())
}

#[test]
fn repeated_segments_use_the_latest_consumer_boundary() -> VortexResult<()> {
    let layout = chunks(vec![flat(0, 20), flat(1, 20), flat(0, 20)]);
    let lifetimes = read_lifetimes(&layout, &locations(2))?;
    assert_eq!(
        lifetimes[&IoTarget::range(0, 4)],
        ReadLifetime::ThroughRow(60)
    );
    assert_eq!(
        lifetimes[&IoTarget::range(4, 4)],
        ReadLifetime::ThroughRow(40)
    );
    Ok(())
}

#[test]
fn missing_segment_locations_are_rejected() {
    let error = read_lifetimes(&flat(1, 20), &locations(1)).expect_err("segment 1 is absent");
    assert!(error.to_string().contains("no known location"));
}
