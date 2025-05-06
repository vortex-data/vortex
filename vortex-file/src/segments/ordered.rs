use std::collections::{BTreeMap, BTreeSet};
use std::task::Waker;

use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_layout::segments::SegmentId;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Debug)]
pub(super) struct Section(Vec<usize>);

impl Default for Section {
    fn default() -> Self {
        Section(vec![0])
    }
}

impl Section {
    pub fn subsection(&self, idx: usize) -> Self {
        let mut ordinals = self.0.clone();
        ordinals.push(idx);
        Section(ordinals)
    }

    pub fn increment(&mut self) {
        let last = self.0.last_mut().vortex_expect("must have section id");
        *last += 1;
    }

    pub fn split(&self, splits: usize, starting_from: usize) -> impl Iterator<Item = Self> {
        (starting_from..splits + starting_from).map(|idx| self.subsection(idx))
    }
}

pub(super) struct OrderedBuffers {
    data: BTreeMap<Section, Vec<ByteBuffer>>,
    active_sections: BTreeSet<Section>,
    wakers: HashMap<Section, Waker>,
    next_segment_id: SegmentId,
}

impl Default for OrderedBuffers {
    fn default() -> Self {
        Self {
            data: Default::default(),
            active_sections: [Section::default()].into(),
            wakers: Default::default(),
            next_segment_id: Default::default(),
        }
    }
}

impl OrderedBuffers {
    pub fn finish_section(
        &mut self,
        section: &Section,
    ) -> Option<BTreeMap<Section, Vec<ByteBuffer>>> {
        self.active_sections.remove(section);
        let Ok(first) = self.first_section() else {
            // last section finished, all is completed
            assert!(self.wakers.is_empty(), "all wakers must have been removed");
            return Some(std::mem::take(&mut self.data));
        };

        if let Some(waker) = self.wakers.remove(&first) {
            waker.wake_by_ref();
        }
        // tail includes incomplete buffers, self.data to only include complete buffers.
        let mut tail = self.data.split_off(&first);
        // swap so tail points to complete buffers, and self.data to incomplete buffers.
        std::mem::swap(&mut tail, &mut self.data);
        Some(tail)
    }

    pub fn split_section(
        &mut self,
        section: &Section,
        splits: usize,
        starting_from: usize,
    ) -> VortexResult<impl Iterator<Item = Section>> {
        if !self.active_sections.remove(&section) {
            vortex_bail!("section not active {:?}", section);
        }
        Ok(section.split(splits, starting_from).map(|section| {
            self.active_sections.insert(section.clone());
            section
        }))
    }

    pub fn add_section(&mut self, section: &Section) {
        self.active_sections.insert(section.clone());
    }

    pub fn insert_buffer(&mut self, idx: Section, buffer: Vec<ByteBuffer>) {
        self.data.insert(idx, buffer);
    }

    pub fn register_waker(&mut self, section: Section, waker: Waker) {
        // TODO(os): should this store a Vec<Waker> instead of replacing?
        self.wakers.insert(section, waker);
    }

    pub fn first_section(&self) -> VortexResult<Section> {
        self.active_sections
            .first()
            .cloned()
            .ok_or_else(|| vortex_err!("no active sections"))
    }

    pub fn next_segment_id(&mut self) -> SegmentId {
        let res = self.next_segment_id;
        self.next_segment_id = SegmentId::from(*res + 1);
        res
    }
}
