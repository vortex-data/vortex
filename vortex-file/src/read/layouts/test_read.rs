use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use croaring::Bitmap;
use itertools::Itertools;
use vortex_array::ArrayData;
use vortex_error::VortexUnwrap;

use crate::read::mask::RowMask;
use crate::{BatchRead, LayoutMessageCache, LayoutReader, MessageLocator};

pub fn layout_splits(layout: &mut dyn LayoutReader, length: usize) -> Vec<RowMask> {
    let mut splits = BTreeSet::new();
    splits.insert(length);
    layout.add_splits(0, &mut splits).vortex_unwrap();
    splits
        .into_iter()
        .tuple_windows::<(usize, usize)>()
        .map(|(begin, end)| unsafe {
            RowMask::new_unchecked(Bitmap::from_range(begin as u32..end as u32), 0, end)
        })
        .collect::<Vec<_>>()
}

pub fn read_layout_data(
    layout: &mut dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    selector: &RowMask,
) -> Option<ArrayData> {
    while let Some(rr) = layout.read_selection(selector).unwrap() {
        match rr {
            BatchRead::ReadMore(m) => {
                let mut write_cache_guard = cache.write().unwrap();
                for MessageLocator(id, range) in m {
                    write_cache_guard.set(id, buf.slice(range.to_range()));
                }
            }
            BatchRead::Batch(a) => return Some(a),
        }
    }
    None
}

pub fn read_filters(
    layout: &mut dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    selector: &RowMask,
) -> Option<RowMask> {
    while let Some(rr) = layout.read_selection(selector).unwrap() {
        match rr {
            BatchRead::ReadMore(m) => {
                let mut write_cache_guard = cache.write().unwrap();
                for MessageLocator(id, range) in m {
                    write_cache_guard.set(id, buf.slice(range.to_range()));
                }
            }
            BatchRead::Batch(a) => {
                return Some(
                    RowMask::from_mask_array(&a, selector.begin(), selector.end()).unwrap(),
                );
            }
        }
    }

    None
}

pub fn filter_read_layout(
    filter_layout: &mut dyn LayoutReader,
    layout: &mut dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    length: usize,
) -> VecDeque<ArrayData> {
    layout_splits(filter_layout, length)
        .into_iter()
        .flat_map(|s| read_filters(filter_layout, cache.clone(), buf, &s))
        .flat_map(|s| read_layout_data(layout, cache.clone(), buf, &s))
        .collect()
}

pub fn read_layout(
    layout: &mut dyn LayoutReader,
    cache: Arc<RwLock<LayoutMessageCache>>,
    buf: &Bytes,
    length: usize,
) -> VecDeque<ArrayData> {
    layout_splits(layout, length)
        .into_iter()
        .flat_map(|s| read_layout_data(layout, cache.clone(), buf, &s))
        .collect()
}
