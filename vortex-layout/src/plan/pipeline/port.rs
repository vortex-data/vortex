// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ports: single-writer, single-reader queues of batches, kept in an arena the scan owns so a
//! port can outlive the stage or split that wrote it.

use std::collections::VecDeque;

use vortex_array::ArrayRef;

/// Identifies a port in the scan's arena.
pub(crate) type PortId = usize;

/// Identifies a pipeline in the scan's table.
pub(crate) type PipelineId = usize;

/// Who reads a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reader {
    /// Nobody: a share's port waiting for the reader it was made for, or a port whose reader
    /// has gone.
    Unclaimed,
    /// A pipeline's source.
    Pipeline(PipelineId),
    /// The scan, for the output of a split's stage.
    Split(usize),
}

/// A queue of batches with one writer and one reader.
pub(crate) struct Queue {
    batches: VecDeque<ArrayRef>,
    /// Batches the queue may hold before its writer is blocked.
    capacity: usize,
    /// The writer will push nothing more.
    closed: bool,
    pub(crate) writer: Option<PipelineId>,
    pub(crate) reader: Reader,
    /// The reader will take nothing more, so pushes are dropped.
    reader_gone: bool,
    /// The slot is free for reuse.
    free: bool,
}

impl Queue {
    fn new(capacity: usize, writer: Option<PipelineId>, reader: Reader) -> Self {
        Self {
            batches: VecDeque::new(),
            capacity: capacity.max(1),
            closed: false,
            writer,
            reader,
            reader_gone: false,
            free: false,
        }
    }

    pub(crate) fn has_room(&self) -> bool {
        self.reader_gone || self.batches.len() < self.capacity
    }

    pub(crate) fn closed(&self) -> bool {
        self.closed
    }

    /// Whether the reader would find something new: a batch, or the close.
    pub(crate) fn readable(&self) -> bool {
        !self.batches.is_empty() || self.closed
    }

    pub(crate) fn pop(&mut self) -> Option<ArrayRef> {
        self.batches.pop_front()
    }
}

/// The reading side of a port, borrowed for one call.
pub struct Inlet<'a> {
    queue: &'a mut Queue,
}

impl Inlet<'_> {
    /// The oldest batch, in place, so the reader can slice it down without removing it.
    pub fn peek_mut(&mut self) -> Option<&mut ArrayRef> {
        self.queue.batches.front_mut()
    }

    /// The oldest batch, removed. Removing a batch is what frees its slot.
    pub fn take(&mut self) -> Option<ArrayRef> {
        self.queue.batches.pop_front()
    }

    /// Whether the writer has finished. Closed and empty is the end of the inlet.
    pub fn closed(&self) -> bool {
        self.queue.closed
    }

    /// Whether no batch is queued.
    pub fn is_empty(&self) -> bool {
        self.queue.batches.is_empty()
    }

    /// Batches queued.
    pub fn len(&self) -> usize {
        self.queue.batches.len()
    }

    /// Whether the inlet has ended: closed with nothing queued.
    pub fn finished(&self) -> bool {
        self.queue.closed && self.queue.batches.is_empty()
    }
}

/// Every port of a scan. Slots are reused once a port's writer has closed it and its reader has
/// gone, so no pipeline still refers to it.
#[derive(Default)]
pub(crate) struct Arena {
    queues: Vec<Queue>,
    free: Vec<PortId>,
}

impl Arena {
    pub(crate) fn create(
        &mut self,
        capacity: usize,
        writer: Option<PipelineId>,
        reader: Reader,
    ) -> PortId {
        let queue = Queue::new(capacity, writer, reader);
        match self.free.pop() {
            Some(id) => {
                self.queues[id] = queue;
                id
            }
            None => {
                self.queues.push(queue);
                self.queues.len() - 1
            }
        }
    }

    pub(crate) fn get(&self, id: PortId) -> &Queue {
        &self.queues[id]
    }

    pub(crate) fn get_mut(&mut self, id: PortId) -> &mut Queue {
        &mut self.queues[id]
    }

    pub(crate) fn inlet(&mut self, id: PortId) -> Inlet<'_> {
        Inlet {
            queue: &mut self.queues[id],
        }
    }

    pub(crate) fn has_room(&self, id: PortId) -> bool {
        self.queues[id].has_room()
    }

    /// Appends a batch, unless the reader has gone. Empty batches are dropped.
    pub(crate) fn push(&mut self, id: PortId, batch: ArrayRef) {
        let queue = &mut self.queues[id];
        if !queue.reader_gone && !batch.is_empty() {
            queue.batches.push_back(batch);
        }
    }

    /// The writer will push nothing more.
    pub(crate) fn close(&mut self, id: PortId) {
        let queue = &mut self.queues[id];
        queue.closed = true;
        queue.writer = None;
        self.maybe_free(id);
    }

    /// The reader will take nothing more: what is queued is dropped, and so is what is pushed.
    pub(crate) fn drop_reader(&mut self, id: PortId) {
        let queue = &mut self.queues[id];
        queue.reader_gone = true;
        queue.reader = Reader::Unclaimed;
        queue.batches.clear();
        self.maybe_free(id);
    }

    fn maybe_free(&mut self, id: PortId) {
        let queue = &mut self.queues[id];
        if queue.closed && queue.reader_gone && !queue.free {
            queue.free = true;
            queue.batches = VecDeque::new();
            self.free.push(id);
        }
    }

    /// Ports in use.
    #[cfg(test)]
    pub(crate) fn live(&self) -> usize {
        self.queues.len() - self.free.len()
    }
}
