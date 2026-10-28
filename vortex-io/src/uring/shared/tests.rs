// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::io;
use std::io::Write;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::unbounded;
use parking_lot::Mutex;
use rstest::rstest;
use rustix::thread::CpuSet;
use rustix::thread::Pid;
use tempfile::NamedTempFile;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::Inner;
use super::SharedUringPool;
use crate::read_at2::ReadId;
use crate::read_at2::ReadRequest;
use crate::read_at2::ReadResult;
use crate::read_at2::ReadTarget;
use crate::uring::FileMode;
use crate::uring::UringFile;
use crate::uring::UringOptions;
use crate::uring::Work;
use crate::uring::driver::Driver;
use crate::uring::tests::TestMode;

#[rstest]
fn same_range_from_independent_producers_uses_one_ring_and_read(
    #[values(1, 2)] rings: usize,
    #[values(TestMode::Buffered, TestMode::Direct, TestMode::Mapped)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    let data: Vec<u8> = (0_u8..251).cycle().take(16384).collect();
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    // Delay reactor intake until every producer has submitted so batching is deterministic.
    let channels: Vec<_> = (0..rings).map(|_| unbounded::<Vec<Work>>()).collect();
    let pool = SharedUringPool(Arc::new(Inner {
        endpoints: channels
            .iter()
            .map(|(sender, _)| Arc::new(Mutex::new(Some(sender.clone()))))
            .collect(),
        workers: Vec::new(),
        check_resident_pages: false,
        tids: Vec::new(),
    }));
    let factory = pool.factory(file.clone());
    let readers = thread::scope(|scope| {
        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let factory = Arc::clone(&factory);
                scope.spawn(move || {
                    let reader = factory()?;
                    reader.submit(&[request(0, 64, 4096)])?;
                    Ok::<_, vortex_error::VortexError>(reader)
                })
            })
            .collect();
        tasks
            .into_iter()
            .map(|task| task.join().expect("producer thread panicked"))
            .collect::<VortexResult<Vec<_>>>()
    })?;
    let batches: Vec<Vec<Work>> = channels
        .iter()
        .map(|(_, receiver)| receiver.try_iter().flatten().collect())
        .collect();
    assert_eq!(batches.iter().filter(|batch| !batch.is_empty()).count(), 1);
    assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), readers.len());
    for batch in batches.into_iter().filter(|batch| !batch.is_empty()) {
        let mut driver = Driver::new(UringOptions {
            reuse_inflight: true,
            ..Default::default()
        })?;
        for work in batch {
            driver.enqueue(work);
        }
        while driver.has_work() {
            driver.pump(true)?;
        }
    }
    assert_eq!(file.stats().logical_reads, readers.len() as u64);
    assert_eq!(file.stats().physical_reads, 1);
    let mut address = None;
    for reader in readers {
        let completion = reader.wait()?;
        assert_eq!(completion.id, ReadId(0));
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.as_ref(), &data[64..4160]);
        assert!(bytes.is_aligned(Alignment::new(64)));
        let current = bytes.as_ptr() as usize;
        assert_eq!(
            *address.get_or_insert(current),
            current,
            "duplicate reads must share their allocation"
        );
        assert!(reader.try_complete()?.is_none());
    }
    Ok(())
}

#[rstest]
fn resident_reads_complete_without_reactor_handoff(
    #[values(1, 2)] rings: usize,
    #[values(
        TestMode::Buffered,
        TestMode::Direct,
        TestMode::Mapped,
        TestMode::MappedRead,
        TestMode::MappedRandom
    )]
    mode: TestMode,
    #[values(false, true)] pinned: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[29; 17003])?;
    let file = mode.open(temporary.path())?;
    let mapped = file.mapping()?.is_some();
    let options = UringOptions {
        check_resident_pages: true,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    };
    let pool = if pinned {
        SharedUringPool::new_pinned(rings, options)?
    } else {
        SharedUringPool::new(rings, options)?
    };
    let reader = pool.factory(file.clone())()?;
    reader.submit(&[request(2, 0, 17003)])?;
    let completion = reader.wait()?;
    assert_eq!(completion.id, ReadId(2));
    completion.result?;
    reader.submit(&[
        request(0, 3, 17000),
        ReadRequest {
            id: ReadId(1),
            target: ReadTarget::Suffix {
                length: 13,
                alignment: Alignment::new(64),
            },
        },
    ])?;
    reader.cancel(ReadId(0));
    let completion = reader.wait()?;
    assert_eq!(completion.id, ReadId(1));
    let ReadResult::Suffix { size, bytes } = completion.result? else {
        panic!("expected suffix")
    };
    assert_eq!(size, 17003);
    let bytes = bytes.try_into_host_sync()?;
    assert_eq!(bytes.as_slice(), &[29; 13]);
    assert!(bytes.is_aligned(Alignment::new(64)));
    assert!(reader.try_complete()?.is_none());
    if mapped {
        assert_eq!(file.stats().physical_reads, 1);
        assert_eq!(file.stats().resident_reads, 2);
        assert_eq!(file.stats().resident_bytes, 17013);
    } else {
        assert!(file.stats().physical_reads > 0);
        assert_eq!(file.stats().resident_reads, 0);
    }
    Ok(())
}

#[rstest]
fn resident_reads_only_skip_populated_ranges_in_the_same_mapping(
    #[values(1, 2)] rings: usize,
    #[values(false, true)] read_mapped_misses: bool,
) -> VortexResult<()> {
    let page = rustix::param::page_size();
    let size = page * 130 + 3;
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![19; size])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let pool = SharedUringPool::new(
        rings,
        UringOptions {
            check_resident_pages: true,
            read_mapped_misses,
            ..Default::default()
        },
    )?;
    let factory = pool.factory(file.clone());
    let reader = factory()?;
    for (id, offset, length, physical_reads) in [
        (0, 0, page * 64, 1),
        (1, page * 63 + 1, page * 2, 2),
        (2, page * 63 + 1, page * 2, 2),
        (3, size - 4, 4, 3),
        (4, size - 4, 4, 3),
    ] {
        reader.submit(&[request(id, offset as u64, length)])?;
        let completion = reader.wait()?;
        assert_eq!(completion.id, ReadId(id));
        let ReadResult::Bytes(bytes) = completion.result? else {
            panic!("expected bytes")
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.len(), length);
        assert!(bytes.iter().all(|&byte| byte == 19));
        assert_eq!(file.stats().physical_reads, physical_reads);
    }
    assert_eq!(file.stats().resident_reads, 2);
    drop(reader);
    assert!(file.release_mapping());
    let reader = factory()?;
    reader.submit(&[request(0, 0, page)])?;
    reader.wait()?.result?;
    assert_eq!(file.stats().physical_reads, 4);
    assert_eq!(file.stats().resident_reads, 2);
    Ok(())
}

fn request(id: u64, offset: u64, length: usize) -> ReadRequest {
    ReadRequest {
        id: ReadId(id),
        target: ReadTarget::Range {
            offset,
            length,
            alignment: Alignment::new(64),
        },
    }
}

#[rstest]
#[case(1, TestMode::Buffered)]
#[case(2, TestMode::Buffered)]
#[case(1, TestMode::Direct)]
#[case(2, TestMode::Direct)]
#[case(1, TestMode::Mapped)]
#[case(2, TestMode::Mapped)]
#[case(1, TestMode::MappedRead)]
#[case(2, TestMode::MappedRead)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_scopes_can_move_between_compute_threads(
    #[case] rings: usize,
    #[case] mode: TestMode,
    #[values(false, true)] force_async: bool,
    #[values(false, true)] pinned: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    // Different 1 MiB windows exercise both reactors when the pool has two rings.
    let window = 1 << 20;
    let data: Vec<u8> = (0_u8..251).cycle().take(2 * window + 17003).collect();
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let options = UringOptions {
        queue_depth: 2,
        reuse_inflight: true,
        force_async,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    };
    let pool = if pinned {
        SharedUringPool::new_pinned(rings, options)?
    } else {
        SharedUringPool::new(rings, options)?
    };
    let factory = pool.factory(file);
    let data = Arc::new(data);
    let mut tasks = Vec::new();
    for i in 0..12 {
        let reader = factory()?;
        let data = Arc::clone(&data);
        let start = (i % 3) * window + (i / 3) * 1024 + 3;
        tasks.push(tokio::spawn(async move {
            reader.submit(&[request(0, 0, 16384)])?;
            reader.cancel(ReadId(0));
            reader.submit(&[request(1, start as u64, 1307)])?;
            tokio::task::yield_now().await;
            let completion = tokio::time::timeout(
                Duration::from_secs(10),
                std::future::poll_fn(|cx| reader.poll_complete(cx)),
            )
            .await
            .map_err(|error| vortex_err!("Completion timeout: {error}"))??;
            assert_eq!(completion.id, ReadId(1));
            let ReadResult::Bytes(bytes) = completion.result? else {
                panic!("expected bytes")
            };
            let bytes = bytes.try_into_host_sync()?;
            assert_eq!(bytes.as_ref(), &data[start..start + 1307]);
            assert!(bytes.is_aligned(Alignment::new(64)));
            assert!(reader.try_complete()?.is_none());
            Ok::<_, vortex_error::VortexError>(())
        }));
    }
    for task in tasks {
        task.await
            .map_err(|error| vortex_err!("Scan task failed: {error}"))??;
    }
    Ok(())
}

#[rstest]
fn blocking_wait_and_registration_failures(
    #[values(false, true)] pinned: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[29; 17003])?;
    let pool = if pinned {
        SharedUringPool::new_pinned(1, UringOptions::default())?
    } else {
        SharedUringPool::new(1, UringOptions::default())?
    };
    let reader = pool.factory(UringFile::open(temporary.path(), FileMode::Buffered)?)()?;
    assert!(
        reader
            .submit(&[request(0, 0, 5), request(1, u64::MAX, 2)])
            .is_err()
    );
    assert!(reader.try_complete()?.is_none());
    assert!(reader.wait().is_err());
    reader.submit(&[
        ReadRequest {
            id: ReadId(0),
            target: ReadTarget::Suffix {
                length: 13,
                alignment: Alignment::new(64),
            },
        },
        request(1, 17002, 2),
    ])?;
    for _ in 0..2 {
        let completion = reader.wait()?;
        if completion.id == ReadId(1) {
            assert!(completion.result.is_err());
        } else {
            let ReadResult::Suffix { size, bytes } = completion.result? else {
                panic!("expected suffix")
            };
            assert_eq!(size, 17003);
            assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[29; 13]);
        }
    }
    // Closing the endpoint must fail accepted registrations instead of leaving a waiter parked.
    pool.0.endpoints[0].lock().take();
    reader.submit(&[request(2, 0, 3)])?;
    assert!(reader.wait()?.result.is_err());
    Ok(())
}

#[rstest]
fn reactor_and_eager_kernel_worker_affinity(
    #[values(1, 2)] rings: usize,
    #[values(false, true)] pinned: bool,
    #[values(false, true)] eager_kernel_worker: bool,
) -> VortexResult<()> {
    let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
    let cpus: Vec<_> = (0..CpuSet::MAX_CPU)
        .filter(|&cpu| allowed.is_set(cpu))
        .collect();
    let options = UringOptions {
        eager_kernel_worker,
        ..Default::default()
    };
    let pool = if pinned {
        SharedUringPool::new_pinned(rings, options)?
    } else {
        SharedUringPool::new(rings, options)?
    };
    assert_eq!(pool.0.tids.len(), rings);
    for (ring, &tid) in pool.0.tids.iter().enumerate() {
        let actual = rustix::thread::sched_getaffinity(Some(tid)).map_err(io::Error::from)?;
        for cpu in 0..CpuSet::MAX_CPU {
            let expected = if pinned {
                cpu == cpus[ring]
            } else {
                allowed.is_set(cpu)
            };
            assert_eq!(actual.is_set(cpu), expected, "reactor {ring}, CPU {cpu}");
        }
        if !eager_kernel_worker {
            continue;
        }
        let owner = tid.as_raw_nonzero().get();
        let expected_name = format!("iou-wrk-{owner}"); // spellchecker:disable-line
        let mut found = false;
        for entry in fs::read_dir("/proc/self/task")? {
            let entry = entry?;
            let name = match fs::read_to_string(entry.path().join("comm")) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                result => result?,
            };
            if name.trim() != expected_name {
                continue;
            }
            let worker = entry
                .file_name()
                .to_string_lossy()
                .parse::<i32>()
                .expect("numeric task directory");
            let worker = Pid::from_raw(worker).expect("positive task id");
            let actual =
                rustix::thread::sched_getaffinity(Some(worker)).map_err(io::Error::from)?;
            for cpu in 0..CpuSet::MAX_CPU {
                assert_eq!(
                    actual.is_set(cpu),
                    allowed.is_set(cpu),
                    "kernel worker CPU {cpu}"
                );
            }
            found = true;
        }
        assert!(
            found,
            "eager setup must create an io-wq worker for reactor {ring}"
        );
    }
    Ok(())
}

#[test]
fn pinned_reactor_count_must_fit_allowed_cpus() -> VortexResult<()> {
    let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
    let cpus = (0..CpuSet::MAX_CPU)
        .filter(|&cpu| allowed.is_set(cpu))
        .count();
    assert!(SharedUringPool::new_pinned(0, UringOptions::default()).is_err());
    assert!(SharedUringPool::new_pinned(cpus + 1, UringOptions::default()).is_err());
    Ok(())
}
