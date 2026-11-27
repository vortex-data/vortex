// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoTarget;
use vortex_mask::Mask;

use super::*;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Split;
use crate::execution::pipeline::synthetic::RowSource;
use crate::execution::pipeline::synthetic::SyntheticPlan;
use crate::execution::pipeline::synthetic::empty_segment;
use crate::plan::ConcatPlan;
use crate::test::SESSION;
use crate::test::Store;

fn locations(entries: &[(u64, u32, usize)]) -> Arc<[SegmentLocation]> {
    entries
        .iter()
        .map(|&(offset, length, align)| SegmentLocation {
            offset,
            length,
            alignment: Alignment::new(align),
        })
        .collect()
}

#[test]
fn empty_scan_yields_and_transitions_to_done() -> VortexResult<()> {
    let plan = RowSource::plan(4, vec![4], None, false);
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..4,
            mask: Mask::new_false(4),
        }],
    )?;
    let mut adaptor = ScanAdaptor::new(scan, Arc::new([]));

    assert_eq!(adaptor.state(), State::NeedsCompute);
    assert!(matches!(adaptor.compute()?, AdaptorStep::Yield));
    assert_eq!(adaptor.state(), State::Done);
    Ok(())
}

#[test]
fn multi_piece_scan_numbers_pieces_sequentially() -> VortexResult<()> {
    let plan = RowSource::plan(5, vec![2, 2, 1], None, false);
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..5,
            mask: Mask::new_true(5),
        }],
    )?;
    let mut adaptor = ScanAdaptor::new(scan, Arc::new([]));

    let mut pieces = Vec::new();
    while adaptor.state() != State::Done {
        match adaptor.compute()? {
            AdaptorStep::Piece(piece) => pieces.push(piece),
            AdaptorStep::Yield => {}
            AdaptorStep::NeedsIO(_) => vortex_bail!("in-memory scan must not request IO"),
        }
    }

    assert_eq!(
        pieces
            .iter()
            .map(|piece| piece.rows.clone())
            .collect::<Vec<_>>(),
        vec![0..1, 1..2, 2..3]
    );
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        pieces[0].array,
        Buffer::from(vec![0u64, 1]).into_array(),
        &mut ctx
    );
    assert_arrays_eq!(
        pieces[1].array,
        Buffer::from(vec![2u64, 3]).into_array(),
        &mut ctx
    );
    assert_arrays_eq!(
        pieces[2].array,
        Buffer::from(vec![4u64]).into_array(),
        &mut ctx
    );
    Ok(())
}

#[test]
fn batches_consecutive_reads_and_retains_pending_output() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    // Chunk 0 reads segment 0, chunk 1 reads segment 1, chunk 2 emits without IO.
    let plan = ConcatPlan::try_new(
        dtype,
        vec![
            RowSource::plan(2, vec![2], Some(SegmentId::from(0)), false),
            RowSource::plan(2, vec![2], Some(SegmentId::from(1)), false),
            RowSource::plan(2, vec![2], None, false),
        ],
    )?
    .into_plan();
    // Split 0 hits the in-memory chunk; split 1 needs segments 0 and 1. Because Scan runs
    // active splits LIFO, split 1 issues both reads first and split 0 immediately emits an
    // output that ScanAdaptor retains in `pending` alongside the NeedsIO batch.
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![
            Split {
                rows: 4..6,
                mask: Mask::new_true(2),
            },
            Split {
                rows: 0..4,
                mask: Mask::new_true(4),
            },
        ],
    )?
    .with_max_active(2);
    let locs = locations(&[(100, 16, 8), (200, 32, 64)]);
    let mut adaptor = ScanAdaptor::new(scan, locs);

    // First compute coalesces both reads into one NeedsIO batch and retains split 0's output.
    let AdaptorStep::NeedsIO(batch) = adaptor.compute()? else {
        vortex_bail!("expected batched IO requests");
    };
    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|req| req.intent == IoIntent::Fetch));
    let req_ids = [batch[0].request, batch[1].request];
    assert_ne!(req_ids[0], req_ids[1]);
    let mut targets: Vec<_> = batch.iter().map(|req| req.target).collect();
    targets.sort_by_key(|target| match target {
        IoTarget::Range { offset, .. } => *offset,
        IoTarget::Size => u64::MAX,
    });
    assert_eq!(
        targets,
        vec![
            IoTarget::Range {
                offset: 100,
                len: 16,
                alignment: Alignment::new(8),
            },
            IoTarget::Range {
                offset: 200,
                len: 32,
                alignment: Alignment::new(64),
            },
        ]
    );
    // Split 0's output is buffered in `pending`, so the adaptor stays NeedsCompute before IO.
    assert_eq!(adaptor.state(), State::NeedsCompute);
    let AdaptorStep::Piece(in_memory_piece) = adaptor.compute()? else {
        vortex_bail!("expected pending output piece before reads complete");
    };
    assert_eq!(in_memory_piece.rows, 0..1);
    assert_arrays_eq!(
        in_memory_piece.array,
        Buffer::from(vec![0u64, 1]).into_array(),
        &mut SESSION.create_execution_ctx()
    );

    // Now nothing is runnable until segments 0 and 1 arrive.
    assert!(matches!(adaptor.compute()?, AdaptorStep::Yield));
    assert_eq!(adaptor.state(), State::Waiting);

    adaptor.set_io_result(req_ids[0], IoResult::Bytes(empty_segment()));
    assert_eq!(adaptor.state(), State::NeedsCompute);
    adaptor.set_io_result(req_ids[1], IoResult::Bytes(empty_segment()));

    let mut remaining = Vec::new();
    while adaptor.state() != State::Done {
        match adaptor.compute()? {
            AdaptorStep::Piece(piece) => remaining.push(piece),
            AdaptorStep::Yield => {}
            AdaptorStep::NeedsIO(_) => vortex_bail!("no further IO expected"),
        }
    }
    assert_eq!(remaining.len(), 2);
    assert_eq!(remaining[0].rows, 1..2);
    assert_eq!(remaining[1].rows, 2..3);
    Ok(())
}

#[test]
fn set_io_result_aligns_delivered_buffer_to_segment_alignment() -> VortexResult<()> {
    struct AlignmentCheckSource {
        requested: bool,
        aligned_ok: Arc<AtomicBool>,
        done: bool,
    }

    impl Source for AlignmentCheckSource {
        fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
            if !self.requested {
                self.requested = true;
                cx.request_read(SegmentId::from(0));
                return Ok(SourceReady::Blocked(SourceBlocked::Io));
            }
            Ok(if self.done || cx.has_bytes() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Io)
            })
        }
    }

    impl Operator for AlignmentCheckSource {
        fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
            if self.done {
                return Ok(Step::End);
            }
            let (_, bytes) = cx
                .take_segment()
                .ok_or_else(|| vortex_err!("missing segment"))?;
            let host = bytes.try_into_host_sync()?;
            self.aligned_ok.store(
                host.as_ptr().addr().is_multiple_of(64) && host.as_ref() == [1, 2, 3, 4],
                Ordering::SeqCst,
            );
            self.done = true;
            Ok(Step::Final(Buffer::from(vec![0u64]).into_array()))
        }
    }

    let aligned_ok = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&aligned_ok);
    let plan = SyntheticPlan::new(
        DType::Primitive(PType::U64, Nullability::NonNullable),
        1,
        Vec::new(),
        move |_, _, _| {
            Ok(Some(Chain::new(AlignmentCheckSource {
                requested: false,
                aligned_ok: Arc::clone(&flag),
                done: false,
            })))
        },
    )
    .into_plan();
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..1,
            mask: Mask::new_true(1),
        }],
    )?;
    let mut adaptor = ScanAdaptor::new(scan, locations(&[(0, 4, 64)]));

    let AdaptorStep::NeedsIO(batch) = adaptor.compute()? else {
        vortex_bail!("expected IO request");
    };
    assert_eq!(adaptor.state(), State::Waiting);

    // Slice at offset 1 so the buffer is intentionally unaligned for 64-byte alignment.
    let unaligned = ByteBuffer::from(vec![99u8, 1, 2, 3, 4]).slice(1..5);
    assert!(!unaligned.as_ptr().addr().is_multiple_of(64));
    adaptor.set_io_result(
        batch[0].request,
        IoResult::Bytes(BufferHandle::new_host(unaligned)),
    );

    assert!(matches!(adaptor.compute()?, AdaptorStep::Piece(_)));
    assert!(aligned_ok.load(Ordering::SeqCst));
    Ok(())
}

#[test]
fn unknown_segment_id_errors_with_context() -> VortexResult<()> {
    let plan = RowSource::plan(2, vec![2], Some(SegmentId::from(5)), false);
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..2,
            mask: Mask::new_true(2),
        }],
    )?;
    let mut adaptor = ScanAdaptor::new(scan, locations(&[(0, 4, 1)]));

    let err = adaptor
        .compute()
        .err()
        .ok_or_else(|| vortex_err!("missing segment location must fail"))?;
    assert!(
        err.to_string()
            .contains("segment SegmentId(5) has no known location"),
        "{err}"
    );
    Ok(())
}

#[test]
fn corrupt_delivered_segment_errors_on_next_compute() -> VortexResult<()> {
    let store = Store::default();
    let plan = store.flat(&Buffer::from(vec![1i32, 2]).into_array())?;
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..2,
            mask: Mask::new_true(2),
        }],
    )?;
    let mut adaptor = ScanAdaptor::new(scan, locations(&[(0, 4, 64)]));

    let AdaptorStep::NeedsIO(batch) = adaptor.compute()? else {
        vortex_bail!("expected IO request");
    };
    adaptor.set_io_result(
        batch[0].request,
        IoResult::Bytes(BufferHandle::new_host(ByteBuffer::from(vec![0u8; 2]))),
    );
    assert_eq!(adaptor.state(), State::NeedsCompute);
    let err = adaptor
        .compute()
        .err()
        .ok_or_else(|| vortex_err!("corrupt segment payload must fail on compute"))?;
    assert!(
        err.to_string()
            .contains("SerializedArray buffer is too short"),
        "{err}"
    );
    Ok(())
}

#[test]
#[should_panic(expected = "which is not outstanding")]
fn set_io_result_panics_on_unknown_request_id() {
    let plan = RowSource::plan(1, vec![1], None, false);
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..1,
            mask: Mask::new_true(1),
        }],
    )
    .expect("scan");
    let mut adaptor = ScanAdaptor::new(scan, Arc::new([]));
    adaptor.set_io_result(IoRequestId(99), IoResult::Bytes(empty_segment()));
}

#[test]
#[should_panic(expected = "answered with a size")]
fn set_io_result_panics_on_size_result() {
    let plan = RowSource::plan(1, vec![1], Some(SegmentId::from(0)), false);
    let scan = Scan::try_new(
        SESSION.clone(),
        plan,
        vec![Split {
            rows: 0..1,
            mask: Mask::new_true(1),
        }],
    )
    .expect("scan");
    let mut adaptor = ScanAdaptor::new(scan, locations(&[(0, 4, 1)]));
    let AdaptorStep::NeedsIO(batch) = adaptor.compute().expect("compute") else {
        panic!("expected NeedsIO");
    };
    adaptor.set_io_result(batch[0].request, IoResult::Size(4));
}
