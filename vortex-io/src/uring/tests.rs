// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use rstest::rstest;
use rustix::thread::CpuSet;
use rustix::thread::Pid;
use tempfile::NamedTempFile;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use super::FileMode;
use super::UringFile;
use super::UringOptions;
use super::UringWorkerPool;
use crate::read_at2::ReadId;
use crate::read_at2::ReadRequest;
use crate::read_at2::ReadResult;
use crate::read_at2::ReadTarget;

mod pool;

#[derive(Clone, Copy, Debug)]
pub(super) enum TestMode {
    Buffered,
    Direct,
    Mapped,
    MappedRead,
    MappedRandom,
}

impl TestMode {
    pub(super) fn open(self, path: &Path) -> VortexResult<UringFile> {
        match self {
            Self::Buffered => UringFile::open(path, FileMode::Buffered),
            Self::Direct => UringFile::open(path, FileMode::Direct),
            // SAFETY: these fixtures are written before opening and never modified afterward.
            Self::Mapped | Self::MappedRead => unsafe { UringFile::open_mapped(path) },
            // SAFETY: these fixtures are written before opening and never modified afterward.
            Self::MappedRandom => unsafe { UringFile::open_mapped_random_access(path) },
        }
    }
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

#[tokio::test]
async fn eager_kernel_worker_inherits_affinity_before_scan_worker_is_pinned() -> VortexResult<()> {
    let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
    let first_cpu = (0..CpuSet::MAX_CPU).find(|&cpu| allowed.is_set(cpu));
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[29; 4096])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let options = UringOptions {
        eager_kernel_worker: true,
        ..Default::default()
    };
    let pool = UringWorkerPool::new(1, options)?;
    pool.spawn(move || {
        let scan_affinity = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
        for cpu in 0..CpuSet::MAX_CPU {
            assert_eq!(scan_affinity.is_set(cpu), Some(cpu) == first_cpu);
        }
        let reader = file.session()?;
        reader.submit(&[request(0, 64, 128)])?;
        let ReadResult::Bytes(bytes) = reader.wait()?.result? else {
            panic!("expected bytes")
        };
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[29; 128]);
        assert_eq!(file.stats().physical_reads, 1);

        let owner = rustix::thread::gettid().as_raw_nonzero().get();
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
            let tid = entry
                .file_name()
                .to_string_lossy()
                .parse::<i32>()
                .expect("numeric task directory");
            let pid = Pid::from_raw(tid).expect("positive task id");
            let actual = rustix::thread::sched_getaffinity(Some(pid)).map_err(io::Error::from)?;
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
            "startup must create an io-wq worker for the scan thread"
        );
        Ok(())
    })
    .await
}

#[rstest]
#[case(TestMode::Buffered, false)]
#[case(TestMode::Buffered, true)]
#[case(TestMode::Direct, false)]
#[case(TestMode::Direct, true)]
#[case(TestMode::Mapped, false)]
#[case(TestMode::Mapped, true)]
#[case(TestMode::MappedRead, false)]
#[case(TestMode::MappedRead, true)]
#[case(TestMode::MappedRandom, false)]
#[case(TestMode::MappedRandom, true)]
#[tokio::test]
async fn exact_bytes_alignment_eof_and_bounded_queue(
    #[case] mode: TestMode,
    #[case] coalesce: bool,
    #[values(false, true)] force_async: bool,
    #[values(false, true)] pinned: bool,
    #[values(false, true)] eager_kernel_worker: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    let data: Vec<u8> = (0_u8..251).cycle().take(17003).collect();
    temporary.write_all(&data)?;
    let file = mode.open(temporary.path())?;
    let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
    let first_cpu = (0..CpuSet::MAX_CPU).find(|&cpu| allowed.is_set(cpu));
    let options = UringOptions {
        queue_depth: 2,
        coalesce,
        force_async,
        eager_kernel_worker,
        read_mapped_misses: matches!(mode, TestMode::MappedRead),
        ..Default::default()
    };
    let pool = if pinned {
        UringWorkerPool::new(1, options)?
    } else {
        UringWorkerPool::with_inherited_affinity(1, options)?
    };
    pool.spawn(move || {
        let actual = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
        for cpu in 0..CpuSet::MAX_CPU {
            assert_eq!(
                actual.is_set(cpu),
                if pinned {
                    Some(cpu) == first_cpu
                } else {
                    allowed.is_set(cpu)
                }
            );
        }
        let reader = file.session()?;
        let requests = [
            request(0, 5, 97),
            request(1, 102, 45),
            request(2, 16600, 403),
            request(3, 17003, 0),
            request(4, 17000, 4),
            ReadRequest {
                id: ReadId(5),
                target: ReadTarget::Suffix {
                    length: 100,
                    alignment: Alignment::new(64),
                },
            },
            ReadRequest {
                id: ReadId(6),
                target: ReadTarget::Size,
            },
        ];
        reader.submit(&requests)?;
        let mut seen = Vec::new();
        for _ in &requests {
            let completion = reader.wait()?;
            seen.push(completion.id.0);
            match completion.id.0 {
                id @ 0..=3 => {
                    let ReadTarget::Range { offset, length, .. } =
                        requests[usize::try_from(id)?].target
                    else {
                        unreachable!()
                    };
                    let ReadResult::Bytes(bytes) = completion.result? else {
                        panic!("expected bytes")
                    };
                    let bytes = bytes.try_into_host_sync()?;
                    let offset = usize::try_from(offset)?;
                    assert_eq!(bytes.as_ref(), &data[offset..offset + length]);
                    assert!(bytes.is_aligned(Alignment::new(64)));
                }
                4 => assert!(completion.result.is_err()),
                5 => {
                    let ReadResult::Suffix { size, bytes } = completion.result? else {
                        panic!("expected suffix")
                    };
                    assert_eq!(size, data.len() as u64);
                    assert_eq!(
                        bytes.try_into_host_sync()?.as_ref(),
                        &data[data.len() - 100..]
                    );
                }
                6 => assert!(matches!(completion.result?, ReadResult::Size(17003))),
                _ => panic!("unknown completion"),
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5, 6]);
        Ok(())
    })
    .await
}

#[rstest]
#[tokio::test]
async fn abandoned_reads_do_not_affect_the_next_scope(
    #[values(TestMode::Buffered, TestMode::Mapped, TestMode::MappedRead)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&vec![19; 1 << 20])?;
    let file = mode.open(temporary.path())?;
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            read_mapped_misses: matches!(mode, TestMode::MappedRead),
            ..Default::default()
        },
    )?;
    pool.spawn(move || {
        let reader = file.session()?;
        reader.submit(&[request(0, 0, 1 << 20)])?;
        drop(reader.try_complete()?);
        reader.cancel(ReadId(0));
        drop(reader);
        let reader = file.session()?;
        reader.submit(&[request(0, 10, 3)])?;
        let result = reader.wait()?;
        assert_eq!(result.id, ReadId(0));
        let ReadResult::Bytes(bytes) = result.result? else {
            panic!("expected bytes")
        };
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[19; 3]);
        Ok(())
    })
    .await
}

#[rstest]
#[tokio::test]
async fn mapped_buffers_survive_releasing_and_replacing_the_cached_mapping(
    #[values(TestMode::Mapped, TestMode::MappedRandom)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[23; 17003])?;
    let file = mode.open(temporary.path())?;
    let pool = UringWorkerPool::new(1, UringOptions::default())?;
    let retained_file = file.clone();
    let (bytes, mapping) = pool
        .spawn(move || {
            let reader = file.session()?;
            let mapping = file.mapping()?.expect("nonempty mapped file");
            let weak = Arc::downgrade(&mapping);
            reader.submit(&[request(0, 64, 128)])?;
            let ReadResult::Bytes(bytes) = reader.wait()?.result? else {
                panic!("expected bytes")
            };
            let bytes = bytes.try_into_host_sync()?;
            assert_eq!(bytes.as_ptr(), mapping[64..].as_ptr());
            assert_eq!(bytes.as_ref(), &[23; 128]);
            drop(mapping);
            drop(reader);
            Ok((bytes, weak))
        })
        .await?;
    assert!(mapping.upgrade().is_some());
    assert_eq!(bytes.as_ref(), &[23; 128]);
    assert!(retained_file.release_mapping());
    assert!(!retained_file.release_mapping());
    let replacement = retained_file.mapping()?.expect("replacement mapping");
    assert!(!Arc::ptr_eq(
        &mapping.upgrade().expect("buffer retains mapping"),
        &replacement
    ));
    assert_eq!(bytes.as_ref(), &[23; 128]);
    drop(bytes);
    assert!(mapping.upgrade().is_none());
    let replacement_weak = Arc::downgrade(&replacement);
    drop(replacement);
    assert!(replacement_weak.upgrade().is_some());
    assert!(retained_file.release_mapping());
    assert!(replacement_weak.upgrade().is_none());
    assert_eq!(retained_file.stats().physical_reads, 1);
    Ok(())
}

#[rstest]
fn mapped_read_ahead_policy_survives_releasing_mapping(
    #[values(TestMode::Mapped, TestMode::MappedRandom)] mode: TestMode,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[23; 17003])?;
    let file = mode.open(temporary.path())?;
    for _ in 0..2 {
        let mapping = file.mapping()?.expect("nonempty mapped file");
        let address = mapping.as_ptr() as usize;
        let mut contains_mapping = false;
        let mut random_access = None;
        for line in fs::read_to_string("/proc/self/smaps")?.lines() {
            if let Some(flags) = line.strip_prefix("VmFlags:") {
                if contains_mapping {
                    random_access = Some(flags.split_whitespace().any(|flag| flag == "rr"));
                    break;
                }
            } else if let Some((start, end)) = line
                .split_whitespace()
                .next()
                .and_then(|range| range.split_once('-'))
            {
                let start = usize::from_str_radix(start, 16).expect("hexadecimal mapping start");
                let end = usize::from_str_radix(end, 16).expect("hexadecimal mapping end");
                contains_mapping = (start..end).contains(&address);
            }
        }
        assert_eq!(random_access, Some(matches!(mode, TestMode::MappedRandom)));
        assert!(file.release_mapping());
    }
    Ok(())
}

#[test]
fn mapped_file_retains_mapping_until_file_is_closed() -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[23; 128])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let mapping = file.mapping()?.expect("nonempty mapped file");
    let weak = Arc::downgrade(&mapping);
    drop(mapping);
    assert!(weak.upgrade().is_some());
    drop(file);
    assert!(weak.upgrade().is_none());
    Ok(())
}

#[tokio::test]
async fn mapped_empty_file_supports_size_and_empty_reads() -> VortexResult<()> {
    let temporary = NamedTempFile::new()?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let pool = UringWorkerPool::new(1, UringOptions::default())?;
    pool.spawn(move || {
        let reader = file.session()?;
        reader.submit(&[
            request(0, 0, 0),
            ReadRequest {
                id: ReadId(1),
                target: ReadTarget::Size,
            },
            ReadRequest {
                id: ReadId(2),
                target: ReadTarget::Suffix {
                    length: 10,
                    alignment: Alignment::new(64),
                },
            },
        ])?;
        for _ in 0..3 {
            let completion = reader.wait()?;
            match completion.result? {
                ReadResult::Size(size) => assert_eq!(size, 0),
                ReadResult::Bytes(bytes) | ReadResult::Suffix { size: 0, bytes } => {
                    assert!(bytes.try_into_host_sync()?.is_empty());
                }
                ReadResult::Suffix { .. } => panic!("expected an empty suffix"),
            }
        }
        assert_eq!(file.stats().physical_reads, 0);
        Ok(())
    })
    .await
}

#[rstest]
#[tokio::test]
async fn future_reads_are_driven_on_their_owning_worker(
    #[values(1, 4)] max_active_jobs: usize,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[31; 128])?;
    let file = UringFile::open(temporary.path(), FileMode::Buffered)?;
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            max_active_jobs,
            ..Default::default()
        },
    )?;
    pool.spawn_async(move || async move {
        let reader = file.session()?;
        for id in 0..3 {
            reader.submit(&[request(id, id, 7)])?;
            let completion = std::future::poll_fn(|cx| reader.poll_complete(cx)).await?;
            assert_eq!(completion.id, ReadId(id));
            let ReadResult::Bytes(bytes) = completion.result? else {
                panic!("expected bytes")
            };
            assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[31; 7]);
        }
        Ok(())
    })
    .await
}
