// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;

use vortex_array::ArrayRef;

use super::split::SplitId;

/// Maintains ordered emission of split results.
///
/// Results are emitted strictly in [`SplitId`] order. Out-of-order pushes are buffered until
/// all preceding splits have been emitted.
pub(crate) struct OutputQueue {
    next_emit: SplitId,
    total_splits: u32,
    buffer: BTreeMap<SplitId, Option<ArrayRef>>,
}

impl OutputQueue {
    pub(crate) fn new(total_splits: u32) -> Self {
        Self {
            next_emit: SplitId::new(0),
            total_splits,
            buffer: BTreeMap::new(),
        }
    }

    /// Pushes a completed split result into the queue.
    pub(crate) fn push(&mut self, id: SplitId, result: Option<ArrayRef>) {
        self.buffer.insert(id, result);
    }

    /// Drains all contiguous splits that are ready for emission (in order).
    pub(crate) fn drain_ready(&mut self) -> Vec<(SplitId, Option<ArrayRef>)> {
        let mut results = Vec::new();
        while let Some(result) = self.buffer.remove(&self.next_emit) {
            results.push((self.next_emit, result));
            self.next_emit = SplitId::new(self.next_emit.as_u32() + 1);
        }
        results
    }

    /// Returns true if all splits have been emitted.
    pub(crate) fn is_complete(&self) -> bool {
        self.next_emit.as_u32() >= self.total_splits
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;

    use super::*;

    #[test]
    fn test_output_queue_ordering() {
        let mut queue = OutputQueue::new(3);

        let array_a = PrimitiveArray::from_iter([1i32]).into_array();
        let array_b = PrimitiveArray::from_iter([2i32]).into_array();
        let array_c = PrimitiveArray::from_iter([3i32]).into_array();

        // Push out of order: split 2 first
        queue.push(SplitId::new(2), Some(array_c));
        assert!(queue.drain_ready().is_empty());

        // Push split 0
        queue.push(SplitId::new(0), Some(array_a));
        let drained = queue.drain_ready();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, SplitId::new(0));

        // Push split 1 — should drain both 1 and 2
        queue.push(SplitId::new(1), Some(array_b));
        let drained = queue.drain_ready();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, SplitId::new(1));
        assert_eq!(drained[1].0, SplitId::new(2));

        assert!(queue.is_complete());
    }

    #[test]
    fn test_output_queue_with_none() {
        let mut queue = OutputQueue::new(2);

        queue.push(SplitId::new(0), None);
        queue.push(
            SplitId::new(1),
            Some(PrimitiveArray::from_iter([1i32]).into_array()),
        );

        let drained = queue.drain_ready();
        assert_eq!(drained.len(), 2);
        assert!(drained[0].1.is_none());
        assert!(drained[1].1.is_some());
        assert!(queue.is_complete());
    }
}
