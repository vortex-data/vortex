// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;
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
fn mapped_misses_use_owned_buffers_then_resident_views(
    #[values(false, true)] read_mapped_misses: bool,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let data: Vec<u8> = (0_u8..251).cycle().take(page * 2 + 3).collect();
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&data)?;
    let file = TestMode::Mapped.open(temporary.path())?;
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
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    let size = rustix::param::page_size() * 1030;
    temporary.write_all(&vec![43; size])?;
    temporary.as_file().sync_all()?;
    let file = TestMode::Mapped.open(temporary.path())?;
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
    };
    let mut driver = Driver::new(UringOptions {
        reuse_inflight: true,
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
fn failed_reactor_completes_each_request_once_and_drains_owned_buffers(
    #[values(TestMode::Buffered, TestMode::Mapped, TestMode::MappedRead)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[0; 4096])?;
    let file = mode.open(temporary.path())?;
    let mapping = file.mapping()?;
    let state = Arc::new(Mutex::new(CompletionState::default()));
    state.lock().live.extend([ReadId(0), ReadId(1)]);
    let mut driver = Driver::new(UringOptions {
        queue_depth: 1,
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
        });
    }
    driver.dispatch()?;
    driver.fail_all("test failure");
    while driver.has_work() {
        driver.pump(true)?;
    }
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
