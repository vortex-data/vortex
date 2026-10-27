// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Timing and recording of individual object store requests.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use futures::StreamExt;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use parking_lot::Mutex;
use tokio::sync::Semaphore;

/// What a request is for. Free-form so workloads can label their requests precisely.
#[derive(Clone, Debug, Default)]
pub struct Tag {
    pub phase: &'static str,
    pub scenario: String,
    pub kind: &'static str,
    pub scan_id: u32,
    pub file: u32,
    /// Logical segment requests this physical request serves.
    pub segments: u32,
    /// Bytes the segments asked for, excluding gap bytes read to coalesce them.
    pub useful_bytes: u64,
    /// The concurrency limit the workload ran under, for grouping.
    pub concurrency: u32,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub tag: Tag,
    pub offset: u64,
    pub length: u64,
    pub inflight_at_start: u64,
    pub seq: u64,
    pub start_us: u64,
    pub headers_us: u64,
    pub first_byte_us: u64,
    pub end_us: u64,
    pub bytes: u64,
    pub chunks: u32,
    pub error: Option<String>,
}

impl Record {
    pub fn total_us(&self) -> u64 {
        self.end_us - self.start_us
    }
}

pub struct Recorder {
    store: Arc<dyn ObjectStore>,
    epoch: Instant,
    inflight: AtomicU64,
    seq: AtomicU64,
    limit: Semaphore,
    records: Mutex<Vec<Record>>,
}

impl Recorder {
    /// `max_inflight` caps requests in flight across every workload, like a process-wide IO pool.
    pub fn new(store: Arc<dyn ObjectStore>, max_inflight: usize) -> Self {
        Self {
            store,
            epoch: Instant::now(),
            inflight: AtomicU64::new(0),
            seq: AtomicU64::new(0),
            limit: Semaphore::new(max_inflight),
            records: Mutex::new(Vec::new()),
        }
    }

    fn now_us(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    pub fn elapsed_us(&self) -> u64 {
        self.now_us()
    }

    /// Issue a ranged GET and record when headers, the first body chunk and the last byte arrive.
    /// The body is drained and dropped; the cost of copying it is not what we are measuring.
    pub async fn get(&self, path: &Path, range: GetRange, tag: Tag) -> u64 {
        // The semaphore is never closed, so acquiring cannot fail.
        let _permit = self.limit.acquire().await.ok();
        let (offset, length) = match &range {
            GetRange::Bounded(r) => (r.start, r.end - r.start),
            GetRange::Suffix(n) => (u64::MAX, *n),
            GetRange::Offset(o) => (*o, 0),
        };
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let inflight_at_start = self.inflight.fetch_add(1, Ordering::Relaxed);
        let start_us = self.now_us();
        let mut headers_us = start_us;
        let mut first_byte_us = 0;
        let mut bytes = 0u64;
        let mut chunks = 0u32;
        let opts = GetOptions {
            range: Some(range),
            ..Default::default()
        };
        let result: object_store::Result<()> = async {
            let response = self.store.get_opts(path, opts).await?;
            headers_us = self.now_us();
            match response.payload {
                GetResultPayload::Stream(mut stream) => {
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        if chunks == 0 {
                            first_byte_us = self.now_us();
                        }
                        chunks += 1;
                        bytes += chunk.len() as u64;
                    }
                }
                GetResultPayload::File(..) => {
                    first_byte_us = self.now_us();
                    bytes = response.range.end - response.range.start;
                }
            }
            Ok(())
        }
        .await;
        let end_us = self.now_us();
        self.inflight.fetch_sub(1, Ordering::Relaxed);
        if first_byte_us == 0 {
            first_byte_us = end_us;
        }
        self.push(Record {
            tag,
            offset,
            length,
            inflight_at_start,
            seq,
            start_us,
            headers_us,
            first_byte_us,
            end_us,
            bytes,
            chunks,
            error: result.err().map(|e| e.to_string()),
        });
        bytes
    }

    /// HEAD the object, as `ObjectStoreReadAt::size` does when opening a file.
    pub async fn head(&self, path: &Path, tag: Tag) -> Option<u64> {
        // The semaphore is never closed, so acquiring cannot fail.
        let _permit = self.limit.acquire().await.ok();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let inflight_at_start = self.inflight.fetch_add(1, Ordering::Relaxed);
        let start_us = self.now_us();
        let result = self.store.head(path).await;
        let end_us = self.now_us();
        self.inflight.fetch_sub(1, Ordering::Relaxed);
        let size = result.as_ref().ok().map(|m| m.size);
        self.push(Record {
            tag: Tag {
                kind: "head",
                ..tag
            },
            offset: 0,
            length: 0,
            inflight_at_start,
            seq,
            start_us,
            headers_us: end_us,
            first_byte_us: end_us,
            end_us,
            bytes: 0,
            chunks: 0,
            error: result.err().map(|e| e.to_string()),
        });
        size
    }

    fn push(&self, record: Record) {
        self.records.lock().push(record);
    }

    pub fn take_records(&self) -> Vec<Record> {
        std::mem::take(&mut *self.records.lock())
    }
}

pub const CSV_HEADER: &str = "run,phase,scenario,kind,scan_id,file,offset,length,segments,\
useful_bytes,concurrency,inflight_at_start,seq,start_us,headers_us,first_byte_us,end_us,\
bytes,chunks,error";

pub fn csv_row(run: &str, r: &Record) -> String {
    let offset = if r.offset == u64::MAX {
        "-1".to_string()
    } else {
        r.offset.to_string()
    };
    let error = r
        .error
        .as_deref()
        .map(|e| e.replace([',', '\n', '\r', '"'], " "))
        .unwrap_or_default();
    format!(
        "{run},{},{},{},{},{},{offset},{},{},{},{},{},{},{},{},{},{},{},{},{error}",
        r.tag.phase,
        r.tag.scenario,
        r.tag.kind,
        r.tag.scan_id,
        r.tag.file,
        r.length,
        r.tag.segments,
        r.tag.useful_bytes,
        r.tag.concurrency,
        r.inflight_at_start,
        r.seq,
        r.start_us,
        r.headers_us,
        r.first_byte_us,
        r.end_us,
        r.bytes,
        r.chunks,
    )
}
