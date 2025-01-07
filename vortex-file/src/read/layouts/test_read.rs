use std::collections::{BTreeSet, VecDeque};

use bytes::Bytes;
use itertools::Itertools;
use vortex_array::ArrayData;
use vortex_error::VortexUnwrap;

use crate::read::mask::RowMask;
use crate::read::splits::SplitsAccumulator;
use crate::{LayoutMessageCache, LayoutReader, MessageLocator, PollRead};

fn layout_splits(layouts: &[&dyn LayoutReader], length: usize) -> impl Iterator<Item = RowMask> {
    let mut splits = BTreeSet::new();
    for layout in layouts {
        layout.add_splits(0, &mut splits).vortex_unwrap();
    }
    splits.insert(length);

    let iter = SplitsAccumulator::new(splits.into_iter().tuple_windows::<(usize, usize)>(), None);

    iter.into_iter().map(|m| m.unwrap())
}

pub fn read_layout_data(
    layout: &dyn LayoutReader,
    buf: &Bytes,
    selector: &RowMask,
    msgs: LayoutMessageCache,
) -> Option<ArrayData> {
    while let Some(rr) = layout.poll_read(selector, &msgs).unwrap() {
        match rr {
            PollRead::ReadMore(m) => {
                msgs.set_many(
                    m.into_iter()
                        .map(|MessageLocator(id, range)| (id, buf.slice(range.as_range()))),
                );
            }
            PollRead::Value(a) => return Some(a),
        }
    }
    None
}

pub fn read_filters(
    layout: &dyn LayoutReader,
    buf: &Bytes,
    selector: &RowMask,
    msgs: LayoutMessageCache,
) -> Option<RowMask> {
    while let Some(rr) = layout.poll_read(selector, &msgs).unwrap() {
        match rr {
            PollRead::ReadMore(m) => {
                msgs.set_many(
                    m.into_iter()
                        .map(|MessageLocator(id, range)| (id, buf.slice(range.as_range()))),
                );
            }
            PollRead::Value(a) => {
                return Some(RowMask::from_array(&a, selector.begin(), selector.end()).unwrap());
            }
        }
    }

    None
}

pub fn filter_read_layout(
    filter_layout: &dyn LayoutReader,
    layout: &dyn LayoutReader,
    buf: &Bytes,
    length: usize,
    msgs: LayoutMessageCache,
) -> VecDeque<ArrayData> {
    layout_splits(&[filter_layout, layout], length)
        .flat_map(|s| read_filters(filter_layout, buf, &s, msgs.clone()))
        .flat_map(|s| read_layout_data(layout, buf, &s, msgs.clone()))
        .collect()
}

pub fn read_layout(
    layout: &dyn LayoutReader,
    buf: &Bytes,
    length: usize,
    msgs: LayoutMessageCache,
) -> VecDeque<ArrayData> {
    layout_splits(&[layout], length)
        .flat_map(|s| read_layout_data(layout, buf, &s, msgs.clone()))
        .collect()
}
