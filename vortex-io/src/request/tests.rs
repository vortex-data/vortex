// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_utils::aliases::hash_map::HashMap;

use super::*;

#[test]
fn request_id_is_a_map_key() {
    let mut pending: HashMap<IoRequestId, IoTarget> = HashMap::default();
    pending.insert(IoRequestId(0), IoTarget::Size);
    pending.insert(IoRequestId(1), IoTarget::range(8, 16));
    assert_eq!(pending.remove(&IoRequestId(0)), Some(IoTarget::Size));
    assert!(pending.contains_key(&IoRequestId(1)));
    assert!(!pending.contains_key(&IoRequestId(0)));
}

#[test]
fn batches_compare_structurally() {
    let batch: IoBatch = vec![IoRequest {
        intent: IoIntent::Fetch,
        request: IoRequestId(3),
        target: IoTarget::range(0, 4),
    }];
    assert_eq!(batch, batch.clone());
    assert_ne!(batch[0].target, IoTarget::range(0, 5));
    assert_eq!(format!("{:?}", IoRequestId(3)), "IoRequestId(3)");
}

#[test]
fn slot_allocates_fresh_ids_and_hands_back_one_result() {
    let mut slot = IoSlot::default();
    assert!(!slot.is_waiting());
    let batch = slot.issue(IoTarget::Size);
    assert_eq!(batch[0].request, IoRequestId(0));
    assert!(slot.is_waiting());
    slot.deliver(IoRequestId(0), IoResult::Size(9));
    assert!(!slot.is_waiting());
    assert!(matches!(slot.take(), Some(IoResult::Size(9))));
    assert!(slot.take().is_none());
    let batch = slot.issue(IoTarget::range(0, 1));
    assert_eq!(batch[0].request, IoRequestId(1));
}

#[test]
fn ranges_carry_their_alignment() {
    let aligned = IoTarget::Range {
        offset: 0,
        len: 4,
        alignment: Alignment::new(8),
    };
    assert_ne!(aligned, IoTarget::range(0, 4));
    assert!(
        IoResult::Bytes(BufferHandle::new_host(vortex_buffer::ByteBuffer::empty()))
            .matches(&aligned)
    );
}

#[test]
#[should_panic(expected = "not outstanding")]
fn slot_rejects_unknown_ids() {
    let mut slot = IoSlot::default();
    slot.issue(IoTarget::Size);
    slot.deliver(IoRequestId(7), IoResult::Size(1));
}

#[test]
#[should_panic(expected = "answered with bytes")]
fn slot_rejects_mismatched_variants() {
    let mut slot = IoSlot::default();
    slot.issue(IoTarget::Size);
    slot.deliver(
        IoRequestId(0),
        IoResult::Bytes(BufferHandle::new_host(vortex_buffer::ByteBuffer::empty())),
    );
}
