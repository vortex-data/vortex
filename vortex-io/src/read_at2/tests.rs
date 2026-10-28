// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use super::FutureReadAt2;
use super::ReadId;
use super::ReadRequest;
use super::ReadResult;
use super::ReadTarget;
use super::VortexReadAt2;
use crate::VortexReadAt;

fn range(id: u64, offset: u64, length: usize) -> ReadRequest {
    ReadRequest {
        id: ReadId(id),
        target: ReadTarget::Range { offset, length, alignment: Alignment::new(64) },
    }
}

#[tokio::test]
async fn adapter_preserves_identity_errors_alignment_and_suffix() -> VortexResult<()> {
    let reader = FutureReadAt2::new(Arc::new(ByteBuffer::from(vec![1, 2, 3, 4, 5])));
    reader.submit(&[
        range(0, 1, 3), range(1, 100, 2),
        ReadRequest { id: ReadId(2), target: ReadTarget::Size },
        ReadRequest { id: ReadId(3), target: ReadTarget::Suffix { length: 2, alignment: Alignment::new(64) } },
    ])?;
    let mut seen = Vec::new();
    for _ in 0..4 {
        let result = std::future::poll_fn(|cx| reader.poll_complete(cx)).await?;
        seen.push(result.id.0);
        match (result.id.0, result.result) {
            (0, Ok(ReadResult::Bytes(bytes))) => {
                let bytes = bytes.to_host().await;
                assert_eq!(bytes.as_ref(), &[2, 3, 4]);
                assert!(bytes.is_aligned(Alignment::new(64)));
            }
            (1, Err(_)) | (2, Ok(ReadResult::Size(5))) => {}
            (3, Ok(ReadResult::Suffix { size: 5, bytes })) => {
                let bytes = bytes.to_host().await;
                assert_eq!(bytes.as_ref(), &[4, 5]);
                assert!(bytes.is_aligned(Alignment::new(64)));
            }
            _ => panic!("unexpected completion"),
        }
    }
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3]);
    Ok(())
}

#[tokio::test]
async fn cancellation_and_rejected_batches_do_not_submit_reads() -> VortexResult<()> {
    let reader = FutureReadAt2::new(Arc::new(ByteBuffer::from(vec![1, 2, 3])));
    assert!(reader.submit(&[range(1, 0, 1), range(1, 1, 1)]).is_err());
    assert!(reader.try_complete()?.is_none());
    reader.submit(&[range(2, 0, 1), range(3, 1, 1)])?;
    reader.cancel(ReadId(2));
    let result = std::future::poll_fn(|cx| reader.poll_complete(cx)).await?;
    assert_eq!(result.id, ReadId(3));
    assert!(reader.try_complete()?.is_none());
    Ok(())
}

struct Delayed;

impl VortexReadAt for Delayed {
    fn concurrency(&self) -> usize { 2 }
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> { async { Ok(2) }.boxed() }
    fn read_at(&self, offset: u64, _: usize, _: Alignment) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async move {
            if offset == 0 { tokio::time::sleep(std::time::Duration::from_millis(30)).await; }
            Ok(BufferHandle::new_host(ByteBuffer::from(vec![offset as u8])))
        }.boxed()
    }
}

#[tokio::test]
async fn adapter_delivers_in_completion_order_and_wakes() -> VortexResult<()> {
    let reader = FutureReadAt2::new(Arc::new(Delayed));
    reader.submit(&[range(0, 0, 1), range(1, 1, 1)])?;
    let first = std::future::poll_fn(|cx| reader.poll_complete(cx)).await?;
    assert_eq!(first.id, ReadId(1));
    let second = std::future::poll_fn(|cx| reader.poll_complete(cx)).await?;
    assert_eq!(second.id, ReadId(0));
    Ok(())
}
