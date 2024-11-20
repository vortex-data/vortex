use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use vortex_array::ArrayData;
use vortex_error::{vortex_panic, VortexUnwrap};

use crate::read::mask::RowMask;
use crate::read::splits::{FixedSplitIterator, MaskIterator, SplitMask};
use crate::{BatchRead, LayoutMessageCache, LayoutReader, MessageLocator};

fn layout_splits(
    layouts: &[&mut dyn LayoutReader],
    length: usize,
) -> impl Iterator<Item = RowMask> {
    let mut iter = FixedSplitIterator::new(length as u64, None);
    let mut splits = BTreeSet::new();
    for layout in layouts {
        layout.add_splits(0, &mut splits).vortex_unwrap();
    }
    iter.additional_splits(&mut splits).vortex_unwrap();
    iter.map(|m| m.vortex_unwrap()).map(|m| match m {
        SplitMask::ReadMore(_) => vortex_panic!("Will never read more"),
        SplitMask::Mask(m) => m,
    })
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
    layout_splits(&[filter_layout, layout], length)
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
    layout_splits(&[layout], length)
        .flat_map(|s| read_layout_data(layout, cache.clone(), buf, &s))
        .collect()
}
