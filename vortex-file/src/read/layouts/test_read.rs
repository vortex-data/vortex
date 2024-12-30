use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, RwLock};

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
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    selector: &RowMask,
) -> Option<ArrayData> {
    while let Some(rr) = layout.poll_read(selector).unwrap() {
        match rr {
            PollRead::ReadMore(m) => {
                let mut write_cache_guard = cache.write().unwrap();
                for MessageLocator(id, range) in m {
                    write_cache_guard.set(id, buf.slice(range.as_range()));
                }
            }
            PollRead::Value(a) => return Some(a),
        }
    }
    None
}

pub fn read_filters(
    layout: &dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    selector: &RowMask,
) -> Option<RowMask> {
    while let Some(rr) = layout.poll_read(selector).unwrap() {
        match rr {
            PollRead::ReadMore(m) => {
                let mut write_cache_guard = cache.write().unwrap();
                for MessageLocator(id, range) in m {
                    write_cache_guard.set(id, buf.slice(range.as_range()));
                }
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
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    length: usize,
) -> VecDeque<ArrayData> {
    layout_splits(&[filter_layout, layout], length)
        .flat_map(|s| read_filters(filter_layout, cache.clone(), buf, &s))
        .flat_map(|s| read_layout_data(layout, cache.clone(), buf, &s))
        .collect()
}

pub fn read_layout(
    layout: &dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    length: usize,
) -> VecDeque<ArrayData> {
    layout_splits(&[layout], length)
        .flat_map(|s| read_layout_data(layout, cache.clone(), buf, &s))
        .collect()
}
