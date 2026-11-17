// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::sync::Arc;

use futures::executor::block_on;
use futures::future::join_all;
use tempfile::TempDir;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::*;
use crate::VortexReadAt;
use crate::request::IoIntent;
use crate::request::IoOwnerId;
use crate::request::IoRequest;
use crate::request::IoRequestId;
use crate::request::IoResult;
use crate::request::IoService;
use crate::request::IoTarget;
use crate::request::OpenTarget;
use crate::request::Openable;
use crate::request::SourceId;
use crate::request::SourcesIoService;

/// The byte at `offset` of file number `file`.
fn byte(file: usize, offset: usize) -> u8 {
    (offset.wrapping_mul(31) ^ file.wrapping_mul(7)) as u8
}

fn write_file(dir: &Path, file: usize, len: usize) -> VortexResult<String> {
    let path = dir.join(format!("{file}.bin"));
    let bytes: Vec<u8> = (0..len).map(|offset| byte(file, offset)).collect();
    std::fs::write(&path, bytes)?;
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| vortex_err!("non-UTF-8 temp path"))
}

fn open(ring: &UringRing, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
    block_on(UringOpenable::new(ring.clone(), path).open())
}

fn fetch(source: SourceId, id: u32, target: IoTarget) -> IoRequest {
    IoRequest {
        source,
        intent: IoIntent::Fetch,
        request: IoRequestId(id),
        target,
    }
}

#[test]
fn reads_match_the_file() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let path = write_file(dir.path(), 0, 1 << 20)?;
    let ring = UringRing::new(64)?;
    let read = open(&ring, &path)?;
    assert_eq!(block_on(read.size())?, 1 << 20);
    for (offset, len, align) in [(0, 4096, 1), (13, 1000, 64), ((1 << 20) - 7, 7, 8), (5, 0, 1)] {
        let bytes = block_on(read.read_at(offset as u64, len, Alignment::new(align)))?
            .try_to_host_sync()?;
        assert_eq!(bytes.len(), len);
        assert_eq!(bytes.as_ptr() as usize % align, 0);
        let expected: Vec<u8> = (offset..offset + len).map(|o| byte(0, o)).collect();
        assert_eq!(bytes.as_slice(), expected.as_slice());
    }
    Ok(())
}

#[test]
fn reads_of_two_files_go_out_in_one_submit() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let a = write_file(dir.path(), 1, 64 << 10)?;
    let b = write_file(dir.path(), 2, 64 << 10)?;
    let ring = UringRing::new(64)?;
    let (a, b) = (open(&ring, &a)?, open(&ring, &b)?);

    let before = ring.stats();
    let reads: Vec<_> = (0..8)
        .flat_map(|i| {
            let offset = i * 4096;
            [
                (1, offset, a.read_at(offset as u64, 4096, Alignment::none())),
                (2, offset, b.read_at(offset as u64, 4096, Alignment::none())),
            ]
        })
        .collect();
    let queued = ring.stats();
    assert_eq!(queued.queued - before.queued, 16);
    assert_eq!(queued.submits, before.submits, "creating a read submits nothing");

    let (files, reads): (Vec<_>, Vec<_>) = reads
        .into_iter()
        .map(|(file, offset, read)| ((file, offset), read))
        .unzip();
    for ((file, offset), bytes) in files.into_iter().zip(block_on(join_all(reads))) {
        let bytes = bytes?.try_to_host_sync()?;
        let expected: Vec<u8> = (offset..offset + 4096).map(|o| byte(file, o)).collect();
        assert_eq!(bytes.as_slice(), expected.as_slice());
    }
    let after = ring.stats();
    assert_eq!(after.submits - before.submits, 1, "{after:?}");
    assert_eq!(after.submitted - before.submitted, 16);
    Ok(())
}

/// The A1 path end to end: a planner's batch of opens, then a batch of reads naming the
/// opened ids, each one `io_uring_enter` across every file.
#[test]
fn opens_and_reads_through_sources_service_batch_on_one_ring() -> VortexResult<()> {
    const FILES: usize = 50;
    let dir = TempDir::new()?;
    let paths = (0..FILES)
        .map(|file| write_file(dir.path(), file, 8192))
        .collect::<VortexResult<Vec<_>>>()?;
    let ring = UringRing::new(256)?;
    let sources = SourcesIoService::new();
    let session = sources.session();
    let owner = IoOwnerId(1);

    // Every file once, then the first again: the duplicate shares the first's open and id.
    let mut opens: Vec<IoRequest> = paths
        .iter()
        .enumerate()
        .map(|(file, path)| {
            let target = OpenTarget::new(UringOpenable::new(ring.clone(), path.as_str()));
            fetch(sources.id(), file as u32, IoTarget::Open(target))
        })
        .collect();
    opens.push(fetch(
        sources.id(),
        FILES as u32,
        IoTarget::Open(OpenTarget::new(UringOpenable::new(
            ring.clone(),
            paths[0].as_str(),
        ))),
    ));
    let before = ring.stats();
    session.submit(owner, opens)?;
    let mut ids = vec![None; FILES + 1];
    for _ in 0..=FILES {
        let completion = session.wait()?;
        let IoResult::Opened(id) = completion.result? else {
            return Err(vortex_err!("an open was not answered with an id"));
        };
        ids[completion.request.0 as usize] = Some(id);
    }
    let ids: Vec<SourceId> = ids.into_iter().flatten().collect();
    assert_eq!(ids.len(), FILES + 1);
    assert_eq!(ids[0], ids[FILES], "one key, one source");
    assert_eq!(sources.len(), FILES);
    let opened = ring.stats();
    assert_eq!(opened.queued - before.queued, FILES as u64, "deduped by key");
    assert_eq!(opened.submits - before.submits, 1, "{opened:?}");

    let reads = ids[..FILES]
        .iter()
        .enumerate()
        .map(|(file, id)| fetch(*id, file as u32, IoTarget::range(100, 64)))
        .collect();
    session.submit(owner, reads)?;
    for _ in 0..FILES {
        let completion = session.wait()?;
        let file = completion.request.0 as usize;
        let IoResult::Bytes(bytes) = completion.result? else {
            return Err(vortex_err!("a read was not answered with bytes"));
        };
        let expected: Vec<u8> = (100..164).map(|o| byte(file, o)).collect();
        assert_eq!(bytes.try_to_host_sync()?.as_slice(), expected.as_slice());
    }
    let read = ring.stats();
    assert_eq!(read.queued - opened.queued, FILES as u64);
    assert_eq!(read.submits - opened.submits, 1, "{read:?}");
    Ok(())
}

#[test]
fn reading_past_the_end_fails() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let path = write_file(dir.path(), 0, 100)?;
    let ring = UringRing::new(8)?;
    let read = open(&ring, &path)?;
    let err = block_on(read.read_at(90, 20, Alignment::none()))
        .err()
        .ok_or_else(|| vortex_err!("read past the end succeeded"))?;
    assert!(err.to_string().contains("end of file"), "{err}");
    Ok(())
}

#[test]
fn opening_a_missing_file_fails() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let ring = UringRing::new(8)?;
    let missing = dir.path().join("missing").to_string_lossy().into_owned();
    let err = block_on(UringOpenable::new(ring, missing).open())
        .err()
        .ok_or_else(|| vortex_err!("opened a missing file"))?;
    assert!(err.to_string().contains("No such file"), "{err}");
    Ok(())
}

#[test]
fn dropping_operations_in_flight_is_safe() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let path = write_file(dir.path(), 3, 1 << 20)?;
    let ring = UringRing::new(8)?;
    let read = open(&ring, &path)?;
    // More reads than the queue holds, dropped before they are polled, and an open.
    let reads: Vec<_> = (0..32)
        .map(|i| read.read_at(i * 4096, 4096, Alignment::none()))
        .collect();
    let opening = UringOpenable::new(ring.clone(), path.as_str()).open();
    ring.flush()?;
    drop(reads);
    drop(opening);
    let bytes = block_on(read.read_at(0, 16, Alignment::none()))?.try_to_host_sync()?;
    let expected: Vec<u8> = (0..16).map(|o| byte(3, o)).collect();
    assert_eq!(bytes.as_slice(), expected.as_slice());
    // The reaper waits out what is still in flight, then the ring shuts down.
    drop(read);
    drop(ring);
    Ok(())
}

#[test]
fn ring_size_does_not_limit_reads_in_flight() -> VortexResult<()> {
    let dir = TempDir::new()?;
    let path = write_file(dir.path(), 4, 1 << 20)?;
    let ring = UringRing::new(4)?;
    let read = open(&ring, &path)?;
    let reads: Vec<_> = (0..64)
        .map(|i| read.read_at(i * 4096, 4096, Alignment::none()))
        .collect();
    for (i, bytes) in block_on(join_all(reads)).into_iter().enumerate() {
        let bytes = bytes?.try_to_host_sync()?;
        assert_eq!(bytes[0], byte(4, i * 4096));
    }
    Ok(())
}
