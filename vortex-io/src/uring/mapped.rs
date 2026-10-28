// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use memmap2::Advice;
use memmap2::Mmap;
use parking_lot::Mutex;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// Reuses page tables across scopes while the file handle remains open.
pub(super) struct MappedFile {
    current: Mutex<Option<Arc<MappedRegion>>>,
    random_access: bool,
}

impl MappedFile {
    pub(super) fn new(random_access: bool) -> Self {
        Self { current: Mutex::new(None), random_access }
    }

    pub(super) fn get(&self, file: &File, size: u64) -> VortexResult<Option<Arc<MappedRegion>>> {
        if size == 0 {
            return Ok(None);
        }
        let mut current = self.current.lock();
        if let Some(mapping) = current.as_ref() {
            return Ok(Some(Arc::clone(mapping)));
        }
        // SAFETY: both mapped-file constructors require immutability while scopes,
        // outstanding operations or buffers exist.
        let mapping = unsafe { Mmap::map(file) }?;
        if self.random_access {
            mapping.advise(Advice::Random)?;
        }
        let mapping = Arc::new(MappedRegion {
            mapping,
            populated: OnceLock::new(),
        });
        *current = Some(Arc::clone(&mapping));
        Ok(Some(mapping))
    }

    pub(super) fn release(&self) -> bool {
        let mapping = self.current.lock().take();
        let released = mapping.is_some();
        // Unmapping can be expensive. Let a concurrent scope establish its mapping without
        // waiting for the old mapping's page tables to be reclaimed under this mutex.
        drop(mapping);
        released
    }
}

pub(super) struct MappedRegion {
    mapping: Mmap,
    populated: OnceLock<Box<[AtomicU64]>>,
}

impl Deref for MappedRegion {
    type Target = Mmap;

    fn deref(&self) -> &Self::Target { &self.mapping }
}

impl MappedRegion {
    fn page_masks(&self, offset: usize, length: usize) -> Option<impl Iterator<Item = (usize, u64)>> {
        let end = offset.checked_add(length).filter(|&end| end <= self.len())?;
        if length == 0 { return None; }
        let page_size = rustix::param::page_size();
        let first = offset / page_size;
        let end = end.div_ceil(page_size);
        Some((first / 64..end.div_ceil(64)).map(move |word| {
            let start = word * 64;
            let mask = (u64::MAX << first.saturating_sub(start))
                & (u64::MAX >> (64 - (end - start).min(64)));
            (word, mask)
        }))
    }

    pub(super) fn was_populated(&self, offset: usize, length: usize) -> bool {
        // One bit per page, allocated only when the optional fast path is used. This is a
        // history hint, not a residency cache: a hit still requires a fresh mincore check.
        let populated = self.populated.get_or_init(|| {
            (0..self.len().div_ceil(rustix::param::page_size()).div_ceil(64))
                .map(|_| AtomicU64::new(0)).collect()
        });
        self.page_masks(offset, length).is_some_and(|mut masks| {
            masks.all(|(word, mask)| populated[word].load(Ordering::Relaxed) & mask == mask)
        })
    }

    pub(super) fn mark_populated(&self, offset: usize, length: usize) {
        let Some(populated) = self.populated.get() else { return; };
        // The kernel population operation rounds its last page beyond a partial file tail.
        let length = length.min(self.len().saturating_sub(offset));
        if let Some(masks) = self.page_masks(offset, length) {
            for (word, mask) in masks {
                populated[word].fetch_or(mask, Ordering::Relaxed);
            }
        }
    }
}

struct MappedBytes(Arc<MappedRegion>);

impl AsRef<[u8]> for MappedBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.mapping.as_ref()
    }
}

pub(super) fn buffer(mapping: Arc<MappedRegion>) -> ByteBuffer {
    ByteBuffer::from(Bytes::from_owner(MappedBytes(mapping)))
}

/// Checks current OS residency without faulting in missing pages or retaining their contents.
/// Residency is a snapshot; pages can still be reclaimed before the caller uses the buffer.
pub(super) fn is_resident(mapping: &Mmap, offset: usize, length: usize) -> bool {
    let Some(end) = offset.checked_add(length).filter(|&end| end <= mapping.len()) else {
        return false;
    };
    if length == 0 { return true; }
    let page_size = rustix::param::page_size();
    let mut start = offset - offset % page_size;
    let mut residency = [0u8; 1024];
    while start < end {
        let length = (end - start).min(residency.len() * page_size);
        let pages = length.div_ceil(page_size);
        // SAFETY: start is page-aligned and the nonempty range lies in this live mapping.
        // The output has one byte for each page, including a partial final page.
        let result = unsafe {
            libc::mincore(mapping.as_ptr().add(start).cast_mut().cast(), length,
                residency.as_mut_ptr())
        };
        if result != 0 || residency[..pages].iter().any(|&page| page & 1 == 0) {
            return false;
        }
        start += length;
    }
    true
}
