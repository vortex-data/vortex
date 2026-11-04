// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;
use std::num::NonZeroU32;
use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use super::Driver;
use crate::read_at2::ReadId;
use crate::read_at2::ReadResult;
use crate::uring::CompletionState;
use crate::uring::UringOptions;
use crate::uring::Work;
use crate::uring::mapped;
use crate::uring::tests::TestMode;

#[rstest]
fn byte_limit_preserves_progress_and_cancellation(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values((false, false), (true, false), (true, true))] policy: (bool, bool),
    #[values(None, Some(0), Some(2))] canceled: Option<u64>,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let data: Vec<u8> = (0_u8..251).cycle().take(page * 10).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..4).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        max_inflight_bytes: NonZeroUsize::new(page * 2),
        coalesce: policy.0,
        preserve_queue_order: policy.1,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    let requests = [(0, 1), (2, 1), (4, 3), (8, 1)];
    for (id, (offset, length)) in requests.into_iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: (offset * page) as u64,
            length: length * page,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    assert_eq!(driver.active.len(), 2);
    assert_eq!(driver.queued.len(), 2);
    if let Some(id) = canceled {
        state.lock().live.remove(&ReadId(id));
    }
    assert_eq!(driver.active_bytes, (page * 2) as u64);
    while driver.has_work() {
        driver.pump(true)?;
        assert_eq!(
            driver.active_bytes,
            driver
                .active
                .values()
                .map(|op| op.length as u64)
                .sum::<u64>()
        );
        assert!(driver.active_bytes <= (page * 2) as u64 || driver.active.len() == 1);
    }
    assert_eq!(driver.active_bytes, 0);
    let results: Vec<_> = state.lock().ready.drain(..).collect();
    assert_eq!(results.len(), 4 - usize::from(canceled.is_some()));
    for completion in results {
        assert_ne!(Some(completion.id.0), canceled);
        let (offset, length) = requests[usize::try_from(completion.id.0)?];
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(
            bytes.try_into_host_sync()?.as_slice(),
            &data[offset * page..(offset + length) * page]
        );
    }
    Ok(())
}

#[rstest]
fn byte_limit_counts_aligned_ranges(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(false, true)] preserve_queue_order: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![37; rustix::param::page_size() * 8])?;
    let file = mode.open(temporary.path())?;
    let block = if matches!(mode, TestMode::MappedRead) {
        1
    } else {
        file.0.offset_alignment
    };
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..2).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        max_inflight_bytes: NonZeroUsize::new(block * 2),
        preserve_queue_order,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    for id in 0..2 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: (block as u64) * (1 + id * 4) - 1,
            length: 2,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    assert_eq!(driver.active.len(), 1);
    assert_eq!(driver.active_bytes, (block * 2) as u64);
    assert_eq!(driver.queued.len(), 1);
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(driver.active_bytes, 0);
    assert_eq!(state.lock().ready.len(), 2);
    for completion in state.lock().ready.drain(..) {
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[37, 37]);
    }
    Ok(())
}

#[rstest]
fn kernel_worker_limits_preserve_async_read_progress(
    #[values(1, 2, 4)] limit: u32,
    #[values(false, true)] eager: bool,
    #[values(TestMode::Buffered, TestMode::Mapped)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    let data: Vec<u8> = (0_u8..251).cycle().take(65536).collect();
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    let mut driver = Driver::new(UringOptions {
        force_async: true,
        eager_kernel_worker: eager,
        max_kernel_workers: NonZeroU32::new(limit),
        coalesce: false,
        ..Default::default()
    })?;
    state.lock().live.extend((0..16).map(ReadId));
    for id in 0..16 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: id * 4096,
            length: 4096,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    while driver.has_work() {
        driver.pump(true)?;
    }
    let mut registered = [0; 2];
    driver
        .ring
        .submitter()
        .register_iowq_max_workers(&mut registered)?;
    assert_eq!(registered, [limit; 2]);
    let results: Vec<_> = state.lock().ready.drain(..).collect();
    assert_eq!(results.len(), 16);
    for completion in results {
        let start = usize::try_from(completion.id.0)? * 4096;
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        assert_eq!(
            bytes.try_into_host_sync()?.as_slice(),
            &data[start..start + 4096]
        );
    }
    Ok(())
}

#[test]
fn byte_limit_releases_capacity_after_short_read_error() -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[37; 4096])?;
    let file = TestMode::Buffered.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let mut driver = Driver::new(UringOptions {
        max_inflight_bytes: NonZeroUsize::new(8192),
        coalesce: false,
        ..Default::default()
    })?;
    for (id, length) in [(0, 8192), (1, 4096)] {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: None,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    assert_eq!(driver.active.len(), 1);
    while driver.has_work() {
        driver.pump(true)?;
        assert_eq!(
            driver.active_bytes,
            driver
                .active
                .values()
                .map(|op| op.length as u64)
                .sum::<u64>()
        );
    }
    assert_eq!(driver.active_bytes, 0);
    assert_eq!(file.stats().physical_reads, 3);
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 2);
    for completion in state.ready.drain(..) {
        if completion.id == ReadId(0) {
            assert!(completion.result.is_err());
        } else {
            let ReadResult::Bytes(bytes) = completion.result? else {
                panic!("expected bytes");
            };
            assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[37; 4096]);
        }
    }
    Ok(())
}

#[rstest]
fn identical_mapped_reads_share_across_rings_until_completion(
    #[values(None, Some(0), Some(1))] canceled: Option<u64>,
    #[values(false, true)] read_mapped_misses: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![37; 8192])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..3).map(ReadId));
    let work = |id| Work {
        file: Arc::clone(&file.0),
        mapping: file.mapping().expect("mapping"),
        state: Arc::clone(&state),
        id: ReadId(id),
        offset: 3,
        length: 4096,
        alignment: if id == 0 {
            Alignment::none()
        } else {
            Alignment::new(64)
        },
        suffix: false,
        inflight: None,
    };
    let options = UringOptions {
        reuse_across_workers: true,
        read_mapped_misses,
        ..Default::default()
    };
    let mut first = Driver::new(options)?;
    let mut second = Driver::new(options)?;
    first.enqueue(work(0));
    second.enqueue(work(1));
    assert!(!second.has_work());
    if let Some(id) = canceled {
        state.lock().live.remove(&ReadId(id));
    }
    while first.has_work() {
        first.pump(true)?;
    }
    let results: Vec<_> = state.lock().ready.drain(..).collect();
    assert_eq!(results.len(), if canceled.is_some() { 1 } else { 2 });
    let mut held = Vec::new();
    for completion in results {
        assert_ne!(Some(completion.id.0), canceled);
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.as_slice(), &[37; 4096]);
        if completion.id == ReadId(1) {
            assert!(bytes.is_aligned(Alignment::new(64)));
        }
        held.push(bytes);
    }
    assert_eq!(file.stats().physical_reads, 1);
    assert_eq!(file.0.inflight.get().expect("registry").pending_count(), 0);
    second.enqueue(work(2));
    while second.has_work() {
        second.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 2);
    assert!(held.iter().all(|bytes| bytes.as_slice() == [37; 4096]));
    Ok(())
}

#[test]
fn shared_reads_keep_replaced_mappings_independent() -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[47; 4096])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let options = UringOptions {
        reuse_across_workers: true,
        ..Default::default()
    };
    let mut first = Driver::new(options)?;
    let mut second = Driver::new(options)?;
    for (id, driver) in [(0, &mut first), (1, &mut second)] {
        file.release_mapping();
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length: 4096,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    assert!(first.has_work());
    assert!(second.has_work());
    while first.has_work() {
        first.pump(true)?;
    }
    while second.has_work() {
        second.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 2);
    assert_eq!(file.0.inflight.get().expect("registry").pending_count(), 0);
    Ok(())
}

#[rstest]
fn shared_pending_reads_fail_when_the_owning_ring_stops(
    #[values(false, true)] dispatched: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![41; 8192])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let options = UringOptions {
        reuse_across_workers: true,
        ..Default::default()
    };
    let mut first = Driver::new(options)?;
    let mut second = Driver::new(options)?;
    for (id, driver) in [(0, &mut first), (1, &mut second)] {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length: 4096,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    if dispatched {
        first.dispatch()?;
    }
    drop(first);
    assert!(!second.has_work());
    let results: Vec<_> = state.lock().ready.drain(..).collect();
    assert_eq!(results.len(), 2);
    for completion in results {
        assert!(completion.result.is_err());
    }
    assert_eq!(file.0.inflight.get().expect("registry").pending_count(), 0);
    Ok(())
}

#[rstest]
fn mapped_misses_use_owned_buffers_then_resident_views(
    #[values(false, true)] read_mapped_misses: bool,
    #[values(TestMode::Mapped, TestMode::MappedRandom)] mode: TestMode,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let data: Vec<u8> = (0_u8..251).cycle().take(page * 2 + 3).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?.expect("nonempty mapping");
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let mut driver = Driver::new(UringOptions {
        check_resident_pages: true,
        read_mapped_misses,
        ..Default::default()
    })?;
    for id in 0..2 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: Some(Arc::clone(&mapping)),
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: page as u64,
            length: page + 3,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
        while driver.has_work() {
            driver.pump(true)?;
        }
        let completion = state.lock().ready.pop_front().expect("completed read");
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.as_slice(), &data[page..]);
        assert_eq!(
            bytes.as_ptr() == mapping.as_ptr().wrapping_add(page),
            id == 1 || !read_mapped_misses
        );
        assert_eq!(file.stats().physical_reads, 1);
        assert_eq!(file.stats().resident_reads, id);
    }
    assert_eq!(
        file.stats().physical_bytes,
        if read_mapped_misses {
            (page + 3) as u64
        } else {
            (page * 2) as u64
        }
    );
    Ok(())
}

#[rstest]
fn resident_page_checks_fall_back_after_eviction(
    #[values(false, true)] check_resident_pages: bool,
    #[values(false, true)] read_mapped_misses: bool,
    #[values(TestMode::Mapped, TestMode::MappedRandom)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    let size = rustix::param::page_size() * 1030;
    temporary.write_all(&vec![43; size])?;
    temporary.as_file().sync_all()?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?.expect("nonempty mapping");
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..3).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        check_resident_pages,
        read_mapped_misses,
        ..Default::default()
    })?;
    for id in 0..3 {
        if id != 1 {
            // SAFETY: the fixture is immutable, and no returned buffer is retained here.
            unsafe { mapping.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) }?;
            rustix::fs::fadvise(&file.0.file, 0, None, rustix::fs::Advice::DontNeed)
                .map_err(io::Error::from)?;
            assert!(!mapped::is_resident(&mapping, 3, size - 3));
        } else {
            assert!(mapped::is_resident(&mapping, 3, size - 3));
        }
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: Some(Arc::clone(&mapping)),
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 3,
            length: size - 3,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
        while driver.has_work() {
            driver.pump(true)?;
        }
        let completion = state.lock().ready.pop_front().expect("completed read");
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.len(), size - 3);
        assert!(bytes.iter().all(|&byte| byte == 43));
        assert!(bytes.is_aligned(Alignment::new(64)));
    }
    assert_eq!(
        file.stats().physical_reads,
        if check_resident_pages { 2 } else { 3 }
    );
    assert_eq!(file.stats().resident_reads, u64::from(check_resident_pages));
    assert!(mapped::is_resident(&mapping, size, 0));
    assert!(!mapped::is_resident(&mapping, size, 1));
    assert!(!mapped::is_resident(&mapping, usize::MAX, 1));
    Ok(())
}

#[rstest]
fn shared_reads_reuse_pending_and_inflight_ranges_but_not_completed_data(
    #[values(TestMode::Buffered, TestMode::Mapped, TestMode::MappedRead)] mode: TestMode,
    #[values(false, true)] preserve_queue_order: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![37; 2 << 20])?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..4).map(ReadId));
    let work = |id, offset, length| Work {
        file: Arc::clone(&file.0),
        mapping: mapping.clone(),
        state: Arc::clone(&state),
        id: ReadId(id),
        offset,
        length,
        alignment: Alignment::new(64),
        suffix: false,
        inflight: None,
    };
    let mut driver = Driver::new(UringOptions {
        reuse_inflight: true,
        preserve_queue_order,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    driver.enqueue(work(0, 0, 2 << 20));
    driver.enqueue(work(1, 0, 2 << 20));
    driver.dispatch()?;
    assert_eq!(driver.active.len(), 1);
    // The SQE owns the buffer now, so a contained request must join it without reallocating it.
    driver.enqueue(work(2, 13, 101));
    state.lock().live.remove(&ReadId(0));
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 1);
    let mut state_guard = state.lock();
    assert_eq!(state_guard.ready.len(), 2);
    for completion in state_guard.ready.drain(..) {
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(
            bytes.len(),
            if completion.id == ReadId(1) {
                2 << 20
            } else {
                101
            }
        );
        assert!(bytes.iter().all(|&byte| byte == 37));
        assert!(bytes.is_aligned(Alignment::new(64)));
    }
    drop(state_guard);
    driver.enqueue(work(3, 13, 101));
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 2);
    Ok(())
}

#[rstest]
fn coalescing_gaps_preserve_read_contents_and_span_limits(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(false, true)] preserve_queue_order: bool,
    #[values(0, 62, 63, usize::MAX)] gap: usize,
    #[values(448, 449)] max_span: usize,
) -> VortexResult<()> {
    let data: Vec<u8> = (0_u8..251).cycle().take(4096).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let mut driver = Driver::new(UringOptions {
        preserve_queue_order,
        max_coalescing_gap: gap,
        max_coalesced_bytes: max_span,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    let ranges = [(0, 257), (320, 129)];
    for (id, (offset, length)) in ranges.into_iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: file.mapping()?,
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: offset as u64,
            length,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    while driver.has_work() {
        driver.pump(true)?;
    }
    let merged = gap >= 63 && max_span == 449;
    assert_eq!(file.stats().physical_reads, if merged { 1 } else { 2 });
    if matches!(mode, TestMode::Buffered | TestMode::MappedRead) {
        assert_eq!(file.stats().physical_bytes, if merged { 449 } else { 386 });
    }
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 2);
    for completion in state.ready.drain(..) {
        let (offset, length) = ranges[usize::try_from(completion.id.0)?];
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(
            bytes.try_into_host_sync()?.as_slice(),
            &data[offset..offset + length]
        );
    }
    Ok(())
}

#[rstest]
fn coalescing_gaps_count_toward_inflight_byte_limit(
    #[values(false, true)] preserve_queue_order: bool,
) -> VortexResult<()> {
    let data: Vec<u8> = (0_u8..251).cycle().take(4096).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = TestMode::Buffered.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..3).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        preserve_queue_order,
        max_coalescing_gap: 64,
        max_inflight_bytes: NonZeroUsize::new(1024),
        ..Default::default()
    })?;
    let ranges = [(0, 256), (320, 256), (2048, 512)];
    for (id, (offset, length)) in ranges.into_iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: None,
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: offset as u64,
            length,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    assert_eq!(driver.active.len(), 1);
    assert_eq!(driver.active_bytes, 576);
    assert_eq!(driver.queued.len(), 1);
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(driver.active_bytes, 0);
    assert_eq!(file.stats().physical_reads, 2);
    assert_eq!(file.stats().physical_bytes, 1088);
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 3);
    for completion in state.ready.drain(..) {
        let (offset, length) = ranges[usize::try_from(completion.id.0)?];
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(
            bytes.try_into_host_sync()?.as_slice(),
            &data[offset..offset + length]
        );
    }
    Ok(())
}

#[rstest]
fn coalescing_dispatches_groups_by_configured_priority(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(false, true)] preserve_queue_order: bool,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let data: Vec<u8> = (0_u8..251).cycle().take(page * 4).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..4).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
        max_coalesced_bytes: page * 2,
        preserve_queue_order,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    let offsets = [2, 0, 3, 1];
    for (id, offset) in offsets.iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: mapping.clone(),
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: (offset * page) as u64,
            length: page,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    let first = driver.active.values().next().expect("first group");
    assert_eq!(
        first.offset,
        if preserve_queue_order {
            (page * 2) as u64
        } else {
            0
        },
    );
    assert_eq!(first.works.len(), 2);
    assert_eq!(driver.queued.len(), 2);
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 2);
    assert_eq!(file.stats().physical_bytes, (page * 4) as u64);
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 4);
    for completion in state.ready.drain(..) {
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        let bytes = bytes.try_into_host_sync()?;
        let offset = offsets[usize::try_from(completion.id.0)?] * page;
        assert_eq!(bytes.as_slice(), &data[offset..offset + page]);
        assert!(bytes.is_aligned(Alignment::new(64)));
    }
    Ok(())
}

#[rstest]
fn fifo_coalescing_keeps_deferred_request_order_after_cancellation(
    #[values(0, 64)] padding: usize,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![37; page * 14])?;
    let file = TestMode::Buffered.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..8).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
        max_coalesced_bytes: page * 2,
        max_coalescing_gap: padding,
        preserve_queue_order: true,
        ..Default::default()
    })?;
    for (id, offset) in [12, 4, 8, 5, 13, 9, 0, 1].into_iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: None,
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: (offset * page) as u64,
            length: page - padding,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    assert_eq!(
        driver
            .queued
            .iter()
            .map(|work| work.id.0)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 5, 6, 7],
    );
    driver.ring.submit_and_wait(1)?;
    driver.reap()?;
    assert!(driver.active.is_empty());
    state.lock().live.remove(&ReadId(1));
    driver.dispatch()?;
    // Cancelling the older request at page 4 must not promote its younger neighbour
    // at page 5 over the request at page 8, which arrived earlier. Gaps between the
    // groups keep cancellation from merging page 5 with that older request.
    assert_eq!(
        driver.active.values().next().expect("second group").offset,
        (page * 8) as u64,
    );
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, 4);
    assert_eq!(state.lock().ready.len(), 7);
    Ok(())
}

#[rstest]
fn fifo_lookahead_preserves_priority_and_read_contents(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(0, 1, 2, 3, 8)] lookahead: usize,
    #[values(false, true)] cancel_oldest: bool,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let data: Vec<_> = (0_u8..251).cycle().take(page * 8).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..8).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
        preserve_queue_order: true,
        coalesce_lookahead: NonZeroUsize::new(lookahead),
        max_coalesced_bytes: page * 2,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    let offsets = [4, 0, 5, 1, 6, 2, 7, 3];
    for (id, offset) in offsets.iter().enumerate() {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: mapping.clone(),
            state: Arc::clone(&state),
            id: ReadId(id as u64),
            offset: (offset * page) as u64,
            length: page,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    if cancel_oldest {
        state.lock().live.remove(&ReadId(0));
    }
    driver.dispatch()?;
    let first = driver.active.values().next().expect("oldest live group");
    assert_eq!(
        first.offset,
        if cancel_oldest { 0 } else { (page * 4) as u64 }
    );
    let merges = lookahead == 0 || lookahead >= 3;
    assert_eq!(first.works.len(), if merges { 2 } else { 1 });
    while driver.has_work() {
        driver.pump(true)?;
    }
    let reads = if cancel_oldest { 7 } else { 8 };
    assert_eq!(file.stats().physical_reads, if merges { 4 } else { reads });
    assert_eq!(file.stats().physical_bytes, reads * page as u64);
    let mut state = state.lock();
    assert_eq!(state.ready.len(), usize::try_from(reads)?);
    for completion in state.ready.drain(..) {
        assert!(!cancel_oldest || completion.id != ReadId(0));
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        let bytes = bytes.try_into_host_sync()?;
        let offset = offsets[usize::try_from(completion.id.0)?] * page;
        assert_eq!(bytes.as_slice(), &data[offset..offset + page]);
    }
    Ok(())
}

#[rstest]
fn fifo_windows_reuse_queued_duplicates_after_dispatch(
    #[values(false, true)] reuse_inflight: bool,
    #[values(None, NonZeroUsize::new(4096))] max_inflight_bytes: Option<NonZeroUsize>,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[37; 4096])?;
    let file = TestMode::Buffered.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..20).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        queue_depth: 2,
        max_inflight_bytes,
        preserve_queue_order: true,
        coalesce_lookahead: NonZeroUsize::new(2),
        reuse_inflight,
        ..Default::default()
    })?;
    for id in 0..20 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: None,
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length: 4096,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(
        file.stats().physical_reads,
        if reuse_inflight { 1 } else { 10 }
    );
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 20);
    for completion in state.ready.drain(..) {
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[37; 4096]);
    }
    Ok(())
}

#[rstest]
fn fifo_rechecks_pages_populated_while_a_request_waits(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(false, true)] recheck: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[37; 4096])?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..2).map(ReadId));
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
        preserve_queue_order: true,
        coalesce_lookahead: NonZeroUsize::new(1),
        check_queued_resident_pages: recheck,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    for id in 0..2 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: mapping.clone(),
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length: 4096,
            alignment: Alignment::new(64),
            suffix: false,
            inflight: None,
        });
    }
    while driver.has_work() {
        driver.pump(true)?;
    }
    let reused = recheck && matches!(mode, TestMode::Mapped | TestMode::MappedRead);
    assert_eq!(file.stats().physical_reads, if reused { 1 } else { 2 });
    assert_eq!(file.stats().resident_reads, u64::from(reused));
    let mut state = state.lock();
    assert_eq!(state.ready.len(), 2);
    for completion in state.ready.drain(..) {
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes");
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[37; 4096]);
    }
    Ok(())
}

#[rstest]
fn failed_reactor_completes_each_request_once_and_drains_owned_buffers(
    #[values(TestMode::Buffered, TestMode::Mapped, TestMode::MappedRead)] mode: TestMode,
    #[values(None, NonZeroUsize::new(4096))] max_inflight_bytes: Option<NonZeroUsize>,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[0; 4096])?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
        max_inflight_bytes,
        coalesce: false,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    })?;
    for id in 0..2 {
        driver.enqueue(Work {
            file: Arc::clone(&file.0),
            mapping: mapping.clone(),
            state: Arc::clone(&state),
            id: ReadId(id),
            offset: 0,
            length: 4096,
            alignment: Alignment::none(),
            suffix: false,
            inflight: None,
        });
    }
    driver.dispatch()?;
    driver.fail_all("test failure");
    while driver.has_work() {
        driver.pump(true)?;
    }
    assert_eq!(driver.active_bytes, 0);
    let state = state.lock();
    assert_eq!(state.ready.len(), 2);
    assert!(
        state
            .ready
            .iter()
            .all(|completion| completion.result.is_err())
    );
    Ok(())
}

#[rstest]
fn live_mapped_buffers_can_be_reused_by_another_ring_until_consumers_drop(
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead
    )]
    mode: TestMode,
    #[values(false, true)] reuse_live_buffers: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![43; 8192])?;
    let file = mode.open(temporary.path())?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend((0..3).map(ReadId));
    let mapping = file.mapping()?;
    let mapped = mapping.is_some();
    let work = |id| Work {
        file: Arc::clone(&file.0),
        mapping: mapping.clone(),
        state: Arc::clone(&state),
        id: ReadId(id),
        offset: 3,
        length: 4096,
        alignment: Alignment::new(64),
        suffix: false,
        inflight: None,
    };
    let options = UringOptions {
        reuse_live_buffers,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    };
    let mut first = Driver::new(options)?;
    let mut second = Driver::new(options)?;
    first.enqueue(work(0));
    while first.has_work() {
        first.pump(true)?;
    }
    let completion = state
        .lock()
        .ready
        .pop_front()
        .expect("first completed read");
    let ReadResult::Bytes(bytes) = completion.result? else {
        panic!("expected bytes")
    };
    let bytes = bytes.try_into_host_sync()?;
    assert_eq!(bytes.as_slice(), &[43; 4096]);
    assert!(bytes.is_aligned(Alignment::new(64)));
    second.enqueue(work(1));
    while second.has_work() {
        second.pump(true)?;
    }
    let expected_reads = if reuse_live_buffers && mapped { 1 } else { 2 };
    assert_eq!(file.stats().physical_reads, expected_reads);
    let completion = state
        .lock()
        .ready
        .pop_front()
        .expect("second completed read");
    let ReadResult::Bytes(reused) = completion.result? else {
        panic!("expected bytes")
    };
    assert_eq!(reused.try_into_host_sync()?.as_slice(), bytes.as_slice());
    drop(bytes);
    second.enqueue(work(2));
    while second.has_work() {
        second.pump(true)?;
    }
    assert_eq!(file.stats().physical_reads, expected_reads + 1);
    Ok(())
}
