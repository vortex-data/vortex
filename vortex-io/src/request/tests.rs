// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_utils::aliases::hash_map::HashMap;

use super::*;

fn segment(id: u32) -> IoTarget {
    IoTarget::segment(SegmentId::from(id))
}

fn empty() -> BufferHandle {
    BufferHandle::new_host(vortex_buffer::ByteBuffer::empty())
}

#[test]
fn request_id_is_a_map_key() {
    let mut pending: HashMap<IoRequestId, IoTarget> = HashMap::default();
    pending.insert(IoRequestId(0), segment(0));
    pending.insert(IoRequestId(1), segment(1));
    assert_eq!(pending.remove(&IoRequestId(0)), Some(segment(0)));
    assert!(pending.contains_key(&IoRequestId(1)));
    assert!(!pending.contains_key(&IoRequestId(0)));
}

#[test]
fn batches_compare_structurally() {
    let batch: IoBatch = vec![IoRequest {
        intent: IoIntent::Fetch,
        request: IoRequestId(3),
        source: SourceId(0),
        target: segment(0),
    }];
    assert_eq!(
        batch,
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(3),
            source: SourceId(0),
            target: segment(0),
        }]
    );
    for request in [
        IoRequest {
            intent: IoIntent::Prefetch,
            ..batch[0].clone()
        },
        IoRequest {
            request: IoRequestId(4),
            ..batch[0].clone()
        },
        IoRequest {
            source: SourceId(1),
            ..batch[0].clone()
        },
        IoRequest {
            target: segment(1),
            ..batch[0].clone()
        },
    ] {
        assert_ne!(batch, vec![request]);
    }
    assert_eq!(format!("{:?}", IoRequestId(3)), "IoRequestId(3)");
}

#[test]
fn slot_allocates_fresh_ids_and_hands_back_one_result() {
    let mut slot = IoSlot::default();
    assert!(!slot.is_waiting());
    let batch = slot.issue(SourceId(0), segment(0));
    assert_eq!(batch[0].request, IoRequestId(0));
    assert!(slot.is_waiting());
    slot.deliver(IoRequestId(0), empty());
    assert!(!slot.is_waiting());
    assert!(slot.take().is_some());
    assert!(slot.take().is_none());
    let batch = slot.issue(SourceId(0), segment(0));
    assert_eq!(batch[0].request, IoRequestId(1));
}

#[test]
fn segment_targets_name_their_bytes() {
    let ranges = IoTarget {
        segment: SegmentId::from(0),
        bytes: SegmentBytes::Ranges(Arc::from([0..4, 8..12])),
    };
    assert_ne!(ranges, segment(0));
}

#[test]
#[should_panic(expected = "not outstanding")]
fn slot_rejects_unknown_ids() {
    let mut slot = IoSlot::default();
    slot.issue(SourceId(0), segment(0));
    slot.deliver(IoRequestId(7), empty());
}
