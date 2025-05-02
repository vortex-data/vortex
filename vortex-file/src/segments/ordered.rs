use std::collections::{BTreeMap, BTreeSet};
use std::task::Waker;

use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_layout::segments::SegmentId;

// [start, end)
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub(super) struct Region {
    pub start: usize,
    pub end: usize,
}

impl Default for Region {
    fn default() -> Self {
        Self {
            start: 0,
            end: usize::MAX,
        }
    }
}

impl Region {
    pub fn split(self, splits: usize) -> VortexResult<impl Iterator<Item = Self>> {
        let step = (self.end - self.start) / splits;
        if step == 0 {
            vortex_bail!("region space exhausted!");
        }
        Ok((self.start..self.end)
            .step_by(step)
            .skip(1)
            .map(move |start| Self {
                start,
                end: start + step,
            }))
    }
}

pub(super) struct OrderedBuffers {
    data: BTreeMap<usize, Vec<ByteBuffer>>,
    active_regions: BTreeSet<Region>,
    wakers: HashMap<Region, Waker>,
    next_segment_id: SegmentId,
}

impl Default for OrderedBuffers {
    fn default() -> Self {
        Self {
            data: Default::default(),
            active_regions: [Region::default()].into(),
            wakers: Default::default(),
            next_segment_id: Default::default(),
        }
    }
}

impl OrderedBuffers {
    pub fn finish_region(&mut self, region: &Region) {
        self.active_regions.remove(&region);
        if let Ok(first) = self.first_region() {
            if let Some(waker) = self.wakers.remove(&first) {
                waker.wake_by_ref();
            }
        }
    }

    pub fn split_region(
        &mut self,
        region: &Region,
        splits: usize,
    ) -> VortexResult<impl Iterator<Item = Region>> {
        if !self.active_regions.remove(&region) {
            vortex_bail!("region not active {:?}", region);
        }
        Ok(region.split(splits)?.map(|region| {
            self.active_regions.insert(region);
            region
        }))
    }

    pub fn insert_buffer(&mut self, idx: usize, buffer: Vec<ByteBuffer>) {
        self.data.insert(idx, buffer);
    }

    pub fn register_waker(&mut self, region: Region, waker: Waker) {
        // TODO(os): should this store a Vec<Waker> instead of replacing?
        self.wakers.insert(region, waker);
    }

    pub fn first_region(&self) -> VortexResult<Region> {
        self.active_regions
            .first()
            .copied()
            .ok_or_else(|| vortex_err!("no active regions"))
    }

    pub fn take_buffers(&mut self) -> VortexResult<BTreeMap<usize, Vec<ByteBuffer>>> {
        if self.active_regions.len() > 1 {
            vortex_bail!("there are more than one active writers");
        }
        if !self.wakers.is_empty() {
            vortex_bail!("there is an inflight write");
        }
        self.active_regions = [Region::default()].into();
        Ok(std::mem::take(&mut self.data))
    }

    pub fn next_segment_id(&mut self) -> SegmentId {
        let res = self.next_segment_id;
        self.next_segment_id = SegmentId::from(*res + 1);
        res
    }
}
