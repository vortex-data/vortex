// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::{bench, Bencher};
use futures::future::join_all;
use futures::Stream;
use futures_util::StreamExt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use tempfile::TempDir;
use tokio::runtime::Runtime;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncReadExt;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexResult, vortex_err};
use vortex_io::source::{IoDriver, IoSource, IoSourceRequest};
use vortex_io::TokioDispatcher;

fn main() {
    divan::main();
}

static DISPATCHER: LazyLock<TokioDispatcher> = LazyLock::new(|| TokioDispatcher::new(1));

/// Helper to create a test file with specified size
fn create_test_file(dir: &TempDir, name: &str, size: usize) -> PathBuf {
    let path = dir.path().join(name);
    let mut file = File::create(&path).expect("Failed to create test file");

    // Write data in chunks for efficiency
    let chunk_size = 1024 * 1024; // 1MB chunks
    let chunk = vec![0xAB; chunk_size];
    let mut written = 0;

    while written < size {
        let to_write = (size - written).min(chunk_size);
        file.write_all(&chunk[..to_write])
            .expect("Failed to write to test file");
        written += to_write;
    }

    file.sync_all().expect("Failed to sync file");
    path
}

// ============================================================================
// Different IoSource implementations to benchmark
// ============================================================================

/// Standard implementation using tokio spawn_blocking with std::fs::File
mod standard_io {
    use super::*;
    use tokio::task::spawn_blocking;
    use vortex_io::Dispatch;

    pub struct StandardFileDriver;

    impl IoDriver for StandardFileDriver {
        type Data = File;

        fn spawn(
            &self,
            requests: impl Stream<Item=IoSourceRequest<Self::Data>> + Send + 'static,
        ) -> VortexResult<()> {
            DISPATCHER.dispatch(move || async move {
                requests
                    .map(move |req| async move {
                        spawn_blocking(move || {
                            let mut buffer =
                                ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                            unsafe { buffer.set_len(req.length) };
                            match req.data().read_exact_at(buffer.as_mut_slice(), req.offset) {
                                Ok(()) => req.resolve(Ok(buffer.freeze())),
                                Err(e) => req.resolve(Err(e.into())),
                            }
                        })
                            .await
                            .expect("Failed to spawn blocking task")
                    })
                    .buffer_unordered(10)
                    .collect::<()>()
                    .await
            })?;
            Ok(())
        }
    }

    pub fn create_source(file: File) -> VortexResult<IoSource> {
        IoSource::try_new(StandardFileDriver, Arc::new(file))
    }
}

/// Implementation using Tokio's async file I/O
mod tokio_async_io {
    use tokio::io::AsyncSeekExt;
    use vortex_io::Dispatch;
    use super::*;

    pub struct TokioAsyncDriver;

    impl IoDriver for TokioAsyncDriver {
        type Data = TokioFile;

        fn spawn(
            &self,
            requests: impl Stream<Item=IoSourceRequest<Self::Data>> + Send + 'static,
        ) -> VortexResult<()> {
            DISPATCHER.dispatch(move || async move {
                requests
                    .map(move |req| async move {
                        let mut buffer =
                            ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                        unsafe { buffer.set_len(req.length) };

                        // Clone the file handle for concurrent reads
                        let mut file = req.data().try_clone().await
                            .expect("Failed to clone file handle");

                        match file.seek(std::io::SeekFrom::Start(req.offset)).await {
                            Ok(_) => {
                                match file.read_exact(buffer.as_mut_slice()).await {
                                    Ok(_) => req.resolve(Ok(buffer.freeze())),
                                    Err(e) => req.resolve(Err(e.into())),
                                }
                            }
                            Err(e) => req.resolve(Err(e.into())),
                        }
                    })
                    .buffer_unordered(10)
                    .collect::<()>()
                    .await
            })?;
            Ok(())
        }
    }

    pub async fn create_source(path: &PathBuf) -> VortexResult<IoSource> {
        let file = TokioFile::open(path).await
            .expect("Failed to open file with Tokio");
        IoSource::try_new(TokioAsyncDriver, Arc::new(file))
    }
}

/// Memory-mapped file implementation using memmap2
mod mmap_io {
    use super::*;
    use memmap2::{Mmap, MmapOptions};
    use vortex_io::Dispatch;

    pub struct MmapDriver;

    pub struct MmapData {
        mmap: Mmap,
    }

    unsafe impl Send for MmapData {}
    unsafe impl Sync for MmapData {}

    impl IoDriver for MmapDriver {
        type Data = MmapData;

        fn spawn(
            &self,
            requests: impl Stream<Item=IoSourceRequest<Self::Data>> + Send + 'static,
        ) -> VortexResult<()> {
            DISPATCHER.dispatch(move || async move {
                requests
                    .map(move |req| async move {
                        let start = req.offset as usize;
                        let end = start + req.length;

                        if end <= req.data().mmap.len() {
                            let slice = &req.data().mmap[start..end];
                            let buffer = ByteBuffer::from(slice.to_vec());
                            req.resolve(Ok(buffer))
                        } else {
                            req.resolve(Err(vortex_err!("Read out of bounds")))
                        }
                    })
                    .buffer_unordered(10)
                    .collect::<()>()
                    .await
            })?;
            Ok(())
        }
    }

    pub fn create_source(file: File) -> VortexResult<IoSource> {
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .expect("Failed to memory map file")
        };
        IoSource::try_new(MmapDriver, Arc::new(MmapData { mmap }))
    }
}

/// Buffered reader implementation with internal caching
mod buffered_io {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tokio::task::spawn_blocking;
    use vortex_io::Dispatch;

    pub struct BufferedDriver;

    pub struct BufferedFile {
        file: File,
        cache: Mutex<HashMap<u64, Arc<ByteBuffer>>>,
        cache_block_size: usize,
    }

    impl BufferedFile {
        pub fn new(file: File) -> Self {
            Self {
                file,
                cache: Mutex::new(HashMap::new()),
                cache_block_size: 256 * 1024, // 256KB blocks
            }
        }

        fn read_block(&self, block_offset: u64) -> VortexResult<Arc<ByteBuffer>> {
            let mut cache = self.cache.lock().unwrap();

            if let Some(cached) = cache.get(&block_offset) {
                return Ok(cached.clone());
            }

            let mut buffer = vec![0u8; self.cache_block_size];
            self.file.read_exact_at(&mut buffer, block_offset)
                .map_err(|e| vortex_err!("Read error: {}", e))?;

            let buffer = Arc::new(ByteBuffer::from(buffer));
            cache.insert(block_offset, buffer.clone());
            Ok(buffer)
        }
    }

    impl IoDriver for BufferedDriver {
        type Data = BufferedFile;

        fn spawn(
            &self,
            requests: impl Stream<Item=IoSourceRequest<Self::Data>> + Send + 'static,
        ) -> VortexResult<()> {
            DISPATCHER.dispatch(move || async move {
                requests
                    .map(move |req| async move {
                        spawn_blocking(move || {
                            let block_size = req.data().cache_block_size;
                            let start_block = (req.offset / block_size as u64) * block_size as u64;
                            let offset_in_block = (req.offset - start_block) as usize;

                            let mut result = ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                            let mut copied = 0;

                            while copied < req.length {
                                let current_block = start_block + ((offset_in_block + copied) / block_size * block_size) as u64;
                                let block_offset = (offset_in_block + copied) % block_size;

                                match req.data().read_block(current_block) {
                                    Ok(block) => {
                                        let to_copy = (req.length - copied).min(block_size - block_offset);
                                        result.extend_from_slice(&block[block_offset..block_offset + to_copy]);
                                        copied += to_copy;
                                    }
                                    Err(e) => {
                                        req.resolve(Err(e));
                                        return;
                                    }
                                }
                            }

                            req.resolve(Ok(result.freeze()))
                        })
                            .await
                            .expect("Failed to spawn blocking task")
                    })
                    .buffer_unordered(10)
                    .collect::<()>()
                    .await
            })?;
            Ok(())
        }
    }

    pub fn create_source(file: File) -> VortexResult<IoSource> {
        IoSource::try_new(BufferedDriver, Arc::new(BufferedFile::new(file)))
    }
}

// ============================================================================
// IoSource factory enum for benchmark arguments
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum IoSourceType {
    Standard,
    TokioAsync,
    Mmap,
    Buffered,
}

impl std::fmt::Display for IoSourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoSourceType::Standard => write!(f, "standard"),
            IoSourceType::TokioAsync => write!(f, "tokio_async"),
            IoSourceType::Mmap => write!(f, "mmap"),
            IoSourceType::Buffered => write!(f, "buffered"),
        }
    }
}

fn create_io_source(source_type: IoSourceType, path: &PathBuf, runtime: &Runtime) -> IoSource {
    match source_type {
        IoSourceType::Standard => {
            let file = File::open(path).expect("Failed to open file");
            standard_io::create_source(file).expect("Failed to create standard source")
        }
        IoSourceType::TokioAsync => {
            runtime.block_on(async {
                tokio_async_io::create_source(path).await
                    .expect("Failed to create tokio async source")
            })
        }
        IoSourceType::Mmap => {
            let file = File::open(path).expect("Failed to open file");
            mmap_io::create_source(file).expect("Failed to create mmap source")
        }
        IoSourceType::Buffered => {
            let file = File::open(path).expect("Failed to open file");
            buffered_io::create_source(file).expect("Failed to create buffered source")
        }
    }
}

// ============================================================================
// Benchmark functions comparing different implementations
// ============================================================================

/// Compare sequential read performance across implementations
#[bench(
    args = [
        (IoSourceType::Standard, 4 * 1024, 500),
        (IoSourceType::TokioAsync, 4 * 1024, 500),
        (IoSourceType::Mmap, 4 * 1024, 500),
        (IoSourceType::Buffered, 4 * 1024, 500),
        
        (IoSourceType::Standard, 64 * 1024, 200),
        (IoSourceType::TokioAsync, 64 * 1024, 200),
        (IoSourceType::Mmap, 64 * 1024, 200),
        (IoSourceType::Buffered, 64 * 1024, 200),
        
        (IoSourceType::Standard, 256 * 1024, 50),
        (IoSourceType::TokioAsync, 256 * 1024, 50),
        (IoSourceType::Mmap, 256 * 1024, 50),
        (IoSourceType::Buffered, 256 * 1024, 50),
        
        (IoSourceType::Standard, 1024 * 1024, 10),
        (IoSourceType::TokioAsync, 1024 * 1024, 10),
        (IoSourceType::Mmap, 1024 * 1024, 10),
        (IoSourceType::Buffered, 1024 * 1024, 10),
    ]
)]
fn sequential_reads(bencher: Bencher, (source_type, read_size, iterations): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 50 * 1024 * 1024; // 50MB file
    let file_path = create_test_file(&temp_dir, "sequential.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                let mut offset = 0u64;
                for _ in 0..iterations {
                    let buffer = source
                        .read(offset, read_size, Alignment::none())
                        .await
                        .expect("Failed to read");

                    assert_eq!(buffer.len(), read_size);
                    offset = (offset + read_size as u64) % (file_size as u64 - read_size as u64);
                }
            });
        });
}

/// Compare random read performance
#[bench(
    args = [
        (IoSourceType::Standard, 4 * 1024, 100),
        (IoSourceType::TokioAsync, 4 * 1024, 100),
        (IoSourceType::Mmap, 4 * 1024, 100),
        (IoSourceType::Buffered, 4 * 1024, 100),
        
        (IoSourceType::Standard, 64 * 1024, 50),
        (IoSourceType::TokioAsync, 64 * 1024, 50),
        (IoSourceType::Mmap, 64 * 1024, 50),
        (IoSourceType::Buffered, 64 * 1024, 50),
        
        (IoSourceType::Standard, 256 * 1024, 20),
        (IoSourceType::TokioAsync, 256 * 1024, 20),
        (IoSourceType::Mmap, 256 * 1024, 20),
        (IoSourceType::Buffered, 256 * 1024, 20),
    ]
)]
fn random_reads(bencher: Bencher, (source_type, read_size, iterations): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 50 * 1024 * 1024; // 50MB file
    let file_path = create_test_file(&temp_dir, "random.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    // Pre-generate random offsets
    let max_offset = file_size - read_size;
    let offsets: Vec<u64> = (0..iterations)
        .map(|i| ((i * 7919) % max_offset) as u64)
        .collect();

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                for &offset in &offsets {
                    let buffer = source
                        .read(offset, read_size, Alignment::none())
                        .await
                        .expect("Failed to read");

                    assert_eq!(buffer.len(), read_size);
                }
            });
        });
}

/// Compare concurrent read performance
#[bench(
    args = [
        (IoSourceType::Standard, 4 * 1024, 20),
        (IoSourceType::TokioAsync, 4 * 1024, 20),
        (IoSourceType::Mmap, 4 * 1024, 20),
        (IoSourceType::Buffered, 4 * 1024, 20),
        
        (IoSourceType::Standard, 64 * 1024, 10),
        (IoSourceType::TokioAsync, 64 * 1024, 10),
        (IoSourceType::Mmap, 64 * 1024, 10),
        (IoSourceType::Buffered, 64 * 1024, 10),
        
        (IoSourceType::Standard, 256 * 1024, 5),
        (IoSourceType::TokioAsync, 256 * 1024, 5),
        (IoSourceType::Mmap, 256 * 1024, 5),
        (IoSourceType::Buffered, 256 * 1024, 5),
    ]
)]
fn concurrent_reads(bencher: Bencher, (source_type, read_size, concurrency): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 50 * 1024 * 1024; // 50MB file
    let file_path = create_test_file(&temp_dir, "concurrent.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                let futures: Vec<_> = (0..concurrency)
                    .map(|i| {
                        let offset = (i * read_size * 2) as u64; // Spread out reads
                        source.read(offset, read_size, Alignment::none())
                    })
                    .collect();

                let results = join_all(futures).await;
                for result in results {
                    let buffer = result.expect("Failed to read");
                    assert_eq!(buffer.len(), read_size);
                }
            });
        });
}

/// Compare performance for small scattered reads (index-like access)
#[bench(
    args = [
        (IoSourceType::Standard, 512, 500),
        (IoSourceType::TokioAsync, 512, 500),
        (IoSourceType::Mmap, 512, 500),
        (IoSourceType::Buffered, 512, 500),
    ]
)]
fn scattered_small_reads(bencher: Bencher, (source_type, read_size, num_reads): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 50 * 1024 * 1024; // 50MB file
    let file_path = create_test_file(&temp_dir, "scattered.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    let offsets: Vec<u64> = (0..num_reads)
        .map(|i| ((i * 49999) % (file_size - read_size)) as u64)
        .collect();

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                for &offset in &offsets {
                    let buffer = source
                        .read(offset, read_size, Alignment::none())
                        .await
                        .expect("Failed to read");

                    assert_eq!(buffer.len(), read_size);
                }
            });
        });
}

/// Compare burst read performance (many reads submitted at once)
#[bench(
    args = [
        (IoSourceType::Standard, 50, 16 * 1024),
        (IoSourceType::TokioAsync, 50, 16 * 1024),
        (IoSourceType::Mmap, 50, 16 * 1024),
        (IoSourceType::Buffered, 50, 16 * 1024),
        
        (IoSourceType::Standard, 100, 4 * 1024),
        (IoSourceType::TokioAsync, 100, 4 * 1024),
        (IoSourceType::Mmap, 100, 4 * 1024),
        (IoSourceType::Buffered, 100, 4 * 1024),
    ]
)]
fn burst_reads(bencher: Bencher, (source_type, burst_size, read_size): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 50 * 1024 * 1024; // 50MB file
    let file_path = create_test_file(&temp_dir, "burst.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                // Submit all reads at once
                let futures: Vec<_> = (0..burst_size)
                    .map(|i| {
                        let offset = ((i * read_size * 2) % (file_size - read_size)) as u64;
                        source.read(offset, read_size, Alignment::none())
                    })
                    .collect();

                // Wait for all to complete
                let results = join_all(futures).await;
                for result in results {
                    let buffer = result.expect("Failed to read");
                    assert_eq!(buffer.len(), read_size);
                }
            });
        });
}

/// Compare large single read performance
#[bench(
    args = [
        (IoSourceType::Standard, 5 * 1024 * 1024),
        (IoSourceType::TokioAsync, 5 * 1024 * 1024),
        (IoSourceType::Mmap, 5 * 1024 * 1024),
        (IoSourceType::Buffered, 5 * 1024 * 1024),
        
        (IoSourceType::Standard, 10 * 1024 * 1024),
        (IoSourceType::TokioAsync, 10 * 1024 * 1024),
        (IoSourceType::Mmap, 10 * 1024 * 1024),
        (IoSourceType::Buffered, 10 * 1024 * 1024),
        
        (IoSourceType::Standard, 20 * 1024 * 1024),
        (IoSourceType::TokioAsync, 20 * 1024 * 1024),
        (IoSourceType::Mmap, 20 * 1024 * 1024),
        (IoSourceType::Buffered, 20 * 1024 * 1024),
    ]
)]
fn large_single_read(bencher: Bencher, (source_type, read_size): (IoSourceType, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 30 * 1024 * 1024; // 30MB file
    let file_path = create_test_file(&temp_dir, "large.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                let buffer = source
                    .read(0, read_size, Alignment::none())
                    .await
                    .expect("Failed to read");

                assert_eq!(buffer.len(), read_size);
            });
        });
}

/// Compare aligned vs unaligned reads
#[bench(
    args = [
        (IoSourceType::Standard, 64 * 1024, 1),
        (IoSourceType::Standard, 64 * 1024, 512),
        (IoSourceType::Standard, 64 * 1024, 4096),
        
        (IoSourceType::TokioAsync, 64 * 1024, 1),
        (IoSourceType::TokioAsync, 64 * 1024, 512),
        (IoSourceType::TokioAsync, 64 * 1024, 4096),
        
        (IoSourceType::Mmap, 64 * 1024, 1),
        (IoSourceType::Mmap, 64 * 1024, 512),
        (IoSourceType::Mmap, 64 * 1024, 4096),
        
        (IoSourceType::Buffered, 64 * 1024, 1),
        (IoSourceType::Buffered, 64 * 1024, 512),
        (IoSourceType::Buffered, 64 * 1024, 4096),
    ]
)]
fn aligned_reads(bencher: Bencher, (source_type, read_size, alignment): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 10 * 1024 * 1024; // 10MB file
    let file_path = create_test_file(&temp_dir, "aligned.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");
    let alignment = Alignment::new(alignment);

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                let iterations = 100;
                for i in 0..iterations {
                    let offset = ((i * read_size) % (file_size - read_size)) as u64;
                    let buffer = source
                        .read(offset, read_size, alignment)
                        .await
                        .expect("Failed to read");

                    assert_eq!(buffer.len(), read_size);
                    // Verify alignment
                    let ptr = buffer.as_ptr() as usize;
                    assert_eq!(ptr % *alignment, 0, "Buffer not properly aligned");
                }
            });
        });
}

/// Test behavior under heavy concurrent load
#[bench(
    args = [
        (IoSourceType::Standard, 100, 4 * 1024),
        (IoSourceType::TokioAsync, 100, 4 * 1024),
        (IoSourceType::Mmap, 100, 4 * 1024),
        (IoSourceType::Buffered, 100, 4 * 1024),
    ]
)]
fn stress_test_concurrent(bencher: Bencher, (source_type, num_concurrent, read_size): (IoSourceType, usize, usize)) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_size = 100 * 1024 * 1024; // 100MB file
    let file_path = create_test_file(&temp_dir, "stress.bin", file_size);

    let runtime = Runtime::new().expect("Failed to create runtime");

    bencher
        .with_inputs(|| create_io_source(source_type, &file_path, &runtime))
        .bench_local_values(|source| {
            runtime.block_on(async {
                let futures: Vec<_> = (0..num_concurrent)
                    .map(|i| {
                        let offset = ((i * 97) % (file_size / read_size)) as u64 * read_size as u64;
                        source.read(offset, read_size, Alignment::none())
                    })
                    .collect();

                let results = join_all(futures).await;
                for result in results {
                    let buffer = result.expect("Failed to read");
                    assert_eq!(buffer.len(), read_size);
                }
            });
        });
}