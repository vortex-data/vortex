// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use futures::future::BoxFuture;
use rstest::rstest;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;

use super::*;
use crate::request::IoRequestId;
use crate::runtime::current::CurrentThreadRuntime;

fn request(intent: IoIntent) -> IoRequest {
    IoRequest {
        intent,
        request: IoRequestId(0),
        target: IoTarget::range(0, 4),
    }
}

#[rstest]
#[case(IoIntent::Announce)]
#[case(IoIntent::Prefetch)]
fn optional_hints_can_be_declined(#[case] intent: IoIntent) -> VortexResult<()> {
    let source = ReadAtIoSource::new(Arc::new(PanickingReadAt), Arc::new(RUNTIME.clone()));
    source.submit(IoOwnerId(1), vec![request(intent)])?;
    assert!(source.poll()?.is_none());
    assert!(source.queued.lock().is_empty());
    Ok(())
}

#[test]
fn registration_does_not_read_and_delivery_keeps_its_identity() -> VortexResult<()> {
    let read = Arc::new(RecordingReadAt::new(buffer![7u8; 4].into_byte_buffer()));
    let source = ReadAtIoSource::new(Arc::clone(&read) as _, Arc::new(RUNTIME.clone()));
    source.submit(IoOwnerId(3), vec![request(IoIntent::Fetch)])?;
    assert!(read.reads().is_empty());
    let completion = source.wait()?;
    assert_eq!(completion.owner, IoOwnerId(3));
    assert_eq!(completion.request, IoRequestId(0));
    assert!(matches!(completion.result?, IoResult::Bytes(bytes) if bytes.len() == 4));
    assert_eq!(read.reads(), vec![(0, 4)]);
    assert!(source.queued.lock().is_empty());
    Ok(())
}

#[test]
fn retirement_and_run_cleanup_cancel_queued_reads() -> VortexResult<()> {
    let source = ReadAtIoSource::new(Arc::new(PanickingReadAt), Arc::new(RUNTIME.clone()));
    source.submit(IoOwnerId(1), vec![request(IoIntent::Fetch)])?;
    source.submit(IoOwnerId(2), vec![request(IoIntent::Fetch)])?;
    source.release(IoOwnerId(1));
    assert_eq!(source.queued.lock().len(), 1);
    assert_eq!(source.queued.lock()[0].0, IoOwnerId(2));
    source.clear();
    assert!(source.queued.lock().is_empty());
    Ok(())
}

#[test]
fn a_range_is_read_at_its_alignment() -> VortexResult<()> {
    let read = Arc::new(RecordingReadAt::new(buffer![7u8; 16].into_byte_buffer()));
    let source = ReadAtIoSource::new(Arc::clone(&read) as _, Arc::new(RUNTIME.clone()));
    let alignment = Alignment::new(8);
    source.submit(
        IoOwnerId(0),
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::Range {
                offset: 3,
                len: 8,
                alignment,
            },
        }],
    )?;
    let IoResult::Bytes(bytes) = source.wait()?.result? else {
        vortex_bail!("expected bytes");
    };
    assert!(bytes.try_into_host_sync()?.is_aligned(alignment));
    assert_eq!(read.alignments(), vec![alignment]);
    Ok(())
}

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

struct PanickingReadAt;

impl VortexReadAt for PanickingReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        panic!("unexpected size request")
    }

    fn read_at(
        &self,
        _offset: u64,
        _len: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        panic!("unexpected read")
    }
}

struct RecordingReadAt {
    buffer: ByteBuffer,
    reads: Mutex<Vec<(u64, usize)>>,
    alignments: Mutex<Vec<Alignment>>,
}

impl RecordingReadAt {
    fn new(buffer: ByteBuffer) -> Self {
        Self {
            buffer,
            reads: Mutex::default(),
            alignments: Mutex::default(),
        }
    }

    fn reads(&self) -> Vec<(u64, usize)> {
        self.reads.lock().clone()
    }

    fn alignments(&self) -> Vec<Alignment> {
        self.alignments.lock().clone()
    }
}

impl VortexReadAt for RecordingReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.buffer.size()
    }

    fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.reads.lock().push((offset, len));
        self.alignments.lock().push(alignment);
        self.buffer.read_at(offset, len, alignment)
    }
}
