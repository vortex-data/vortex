// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::squeue;
use io_uring::types;
use rustix::mm::Advice;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use super::UringOptions;
use super::Work;
use super::mapped;

pub(super) struct Driver {
    ring: IoUring,
    options: UringOptions,
    queued: VecDeque<Work>,
    active: HashMap<u64, Physical>,
    next: u64,
}

struct Physical {
    works: Vec<Work>,
    buffer: Destination,
    offset: u64,
    offset_alignment: usize,
    length: usize,
    initialized: usize,
    required: usize,
    failed: bool,
    in_kernel: bool,
}

enum Destination {
    Read(ByteBufferMut),
    Mapped(Arc<mapped::MappedRegion>),
}

impl Driver {
    pub(super) fn new(options: UringOptions) -> VortexResult<Self> {
        if options.queue_depth == 0 { vortex_bail!("io_uring queue depth must be positive"); }
        let mut ring = IoUring::new(options.queue_depth)?;
        if options.eager_kernel_worker { Self::start_kernel_worker(&mut ring)?; }
        Ok(Self { ring, options, queued: VecDeque::new(),
            active: HashMap::default(), next: 0 })
    }

    fn start_kernel_worker(ring: &mut IoUring) -> VortexResult<()> {
        let entry = opcode::Nop::new().build().flags(squeue::Flags::ASYNC).user_data(u64::MAX);
        // SAFETY: NOP has no referenced buffers or file descriptors. The ring is empty.
        unsafe { ring.submission().push(&entry) }
            .map_err(|_| vortex_err!("io_uring startup queue unexpectedly full"))?;
        loop {
            match ring.submit_and_wait(1) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => { result?; }
            }
            if let Some(completion) = ring.completion().next() {
                vortex_ensure!(completion.user_data() == u64::MAX, "Unexpected io_uring startup completion");
                let result = completion.result();
                if result < 0 { return Err(io::Error::from_raw_os_error(-result).into()); }
                vortex_ensure!(result == 0, "Unexpected io_uring startup result {result}");
                return Ok(());
            }
        }
    }

    pub(super) fn enqueue(&mut self, work: Work) {
        if self.options.check_resident_pages && let Some(bytes) = work.resident_buffer() {
            work.complete(Ok(bytes));
            return;
        }
        if self.options.reuse_live_buffers
            && let Some(buffers) = work.file.live_buffers.get()
            && let Some(bytes) = buffers.get(work.offset, work.length, work.alignment)
        {
            work.complete(Ok(bytes));
            return;
        }
        if self.options.reuse_inflight {
            for operation in self.active.values_mut() {
                if !operation.failed && Arc::ptr_eq(&work.file, &operation.works[0].file)
                    && work.offset >= operation.offset
                    && work.offset + work.length as u64 <= operation.offset + operation.required as u64
                {
                    operation.works.push(work);
                    return;
                }
            }
        }
        self.queued.push_back(work);
    }

    pub(super) fn fail_all(&mut self, reason: &str) {
        for work in self.queued.drain(..) { work.complete(Err(vortex_err!("{reason}"))); }
        for operation in self.active.values_mut() {
            if operation.failed { continue; }
            operation.failed = true;
            for work in &operation.works {
                work.clone().complete(Err(vortex_err!("{reason}")));
            }
        }
    }

    pub(super) fn has_work(&self) -> bool {
        !self.queued.is_empty() || !self.active.is_empty()
    }

    pub(super) fn driven_by_worker_loop(&self) -> bool { self.options.max_active_jobs > 1 }

    fn push(&mut self, id: u64) -> VortexResult<()> {
        let operation = self.active.get_mut(&id).ok_or_else(|| vortex_err!("Missing ring operation"))?;
        let remaining = operation.length - operation.initialized;
        let length = u32::try_from(remaining).map_err(|_| vortex_err!("Physical read exceeds u32"))?;
        let file = &operation.works[0].file;
        let mut entry = match &mut operation.buffer {
            Destination::Read(buffer) => {
                // SAFETY: `active` uniquely owns this stable allocation, with at least `length`
                // spare bytes. No buffer access or drop occurs before the operation's CQE.
                let pointer = unsafe { buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>()
                    .add(operation.initialized) };
                opcode::Read::new(types::Fd(file.file.as_raw_fd()), pointer, length)
                    .offset(operation.offset + operation.initialized as u64).build()
            }
            Destination::Mapped(mapping) => {
                // SAFETY: validated nonempty requests lie within this immutable mapping. Its
                // page-aligned range remains mapped until its CQE, including on cancellation.
                let pointer = unsafe { mapping.as_ptr().add(operation.offset as usize) };
                opcode::Madvise::new(pointer.cast(), length as _, Advice::LinuxPopulateRead as i32)
                    .build()
            }
        }.user_data(id);
        if self.options.force_async { entry = entry.flags(squeue::Flags::ASYNC); }
        // SAFETY: fd and buffer or mapping live in `active` through completion, even on cancellation.
        unsafe { self.ring.submission().push(&entry) }.map_err(|_| vortex_err!("io_uring submission queue unexpectedly full"))?;
        operation.in_kernel = true;
        file.counters.physical_reads.fetch_add(1, Ordering::Relaxed);
        file.counters.physical_bytes.fetch_add(remaining as u64, Ordering::Relaxed);
        Ok(())
    }

    fn dispatch(&mut self) -> VortexResult<()> {
        self.queued.retain(Work::live);
        if self.options.coalesce {
            self.queued.make_contiguous().sort_by_key(|work| (work.file.file.as_raw_fd(), work.offset));
        }
        while self.active.len() < self.options.queue_depth as usize {
            let Some(first) = self.queued.pop_front() else { break; };
            let start = first.offset;
            let mut end = start + first.length as u64;
            let mut alignment = first.alignment.as_usize().max(first.file.memory_alignment);
            // A mapped file's page alignment is required by MADV_POPULATE_READ, not by READ.
            let offset_alignment = if first.mapping.is_some() && self.options.read_mapped_misses {
                1
            } else {
                first.file.offset_alignment
            };
            let block = offset_alignment as u64;
            let mut works = vec![first];
            if self.options.coalesce {
                while self.queued.front().is_some_and(|next| {
                    Arc::ptr_eq(&next.file, &works[0].file) && next.offset <= end
                        && ((next.offset + next.length as u64).max(end) - start <= self.options.max_coalesced_bytes as u64
                            || (self.options.reuse_inflight && next.offset + next.length as u64 <= end))
                }) {
                    if let Some(next) = self.queued.pop_front() {
                        end = end.max(next.offset + next.length as u64);
                        alignment = alignment.max(next.alignment.as_usize());
                        works.push(next);
                    }
                }
            }
            let offset = start - start % block;
            let Some(aligned_end) = end.checked_next_multiple_of(block) else {
                for work in works { work.complete(Err(vortex_err!("Aligned read overflow"))); }
                continue;
            };
            let length = usize::try_from(aligned_end - offset)?;
            if length > i32::MAX as usize {
                for work in works { work.complete(Err(vortex_err!("io_uring read exceeds supported operation size"))); }
                continue;
            }
            let id = self.next;
            self.next = self.next.checked_add(1).ok_or_else(|| vortex_err!("Ring identity overflow"))?;
            let buffer = match &works[0].mapping {
                Some(mapping) if !self.options.read_mapped_misses => Destination::Mapped(Arc::clone(mapping)),
                _ => Destination::Read(ByteBufferMut::with_capacity_aligned(length, Alignment::new(alignment))),
            };
            self.active.insert(id, Physical {
                works, buffer,
                offset, offset_alignment, length, initialized: 0, required: (end - offset) as usize,
                failed: false, in_kernel: false,
            });
            self.push(id)?;
        }
        Ok(())
    }

    fn reap(&mut self) -> VortexResult<bool> {
        let completions: Vec<_> = self.ring.completion().map(|cqe| (cqe.user_data(), cqe.result())).collect();
        let progressed = !completions.is_empty();
        for (id, result) in completions {
            let Some(mut operation) = self.active.remove(&id) else { vortex_bail!("Unknown ring completion {id}"); };
            operation.in_kernel = false;
            if operation.failed { continue; }
            if result < 0 {
                for work in operation.works { work.complete(Err(io::Error::from_raw_os_error(-result).into())); }
                continue;
            }
            if let Destination::Mapped(mapping) = &operation.buffer {
                if result == 0 {
                    mapping.mark_populated(operation.offset as usize, operation.length);
                    self.complete(operation.works, mapped::buffer(Arc::clone(mapping)), 0);
                } else {
                    for work in operation.works {
                        work.complete(Err(vortex_err!("Unexpected mapped-page completion result {result}")));
                    }
                }
                continue;
            }
            operation.initialized += result as usize;
            if operation.initialized < operation.required {
                let file = &operation.works[0].file;
                if result == 0 || !operation.initialized.is_multiple_of(operation.offset_alignment)
                    || !operation.initialized.is_multiple_of(file.memory_alignment) {
                    for work in operation.works {
                        work.complete(Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short positional read").into()));
                    }
                } else {
                    self.active.insert(id, operation);
                    self.push(id)?;
                }
                continue;
            }
            let Destination::Read(mut buffer) = operation.buffer else {
                vortex_bail!("Mapped operation reached read completion");
            };
            // SAFETY: this CQE and prior short-read CQEs initialized every byte in this prefix.
            // There are no outstanding kernel operations referring to this allocation.
            unsafe { buffer.set_len(operation.initialized); }
            if let Some(mapping) = &operation.works[0].mapping {
                // A buffered read warms the OS cache without installing this mapping's PTEs.
                // Future fast-path reads still check current residency before using the mapping.
                mapping.mark_populated(operation.offset as usize, operation.initialized);
            }
            self.complete(operation.works, buffer.freeze(), operation.offset);
        }
        Ok(progressed)
    }

    fn complete(&self, works: Vec<Work>, bytes: ByteBuffer, offset: u64) {
        for work in works {
            if !work.live() { continue; }
            let start = (work.offset - offset) as usize;
            let mut buffer = bytes.slice_unaligned(start..start + work.length).aligned(work.alignment);
            if self.options.reuse_live_buffers && work.mapping.is_some() {
                buffer = work.file.live_buffers.get_or_init(Default::default)
                    .insert(work.offset, buffer);
            }
            work.complete(Ok(buffer));
        }
    }

    pub(super) fn pump(&mut self, wait: bool) -> VortexResult<()> {
        let progressed = self.reap()?;
        self.dispatch()?;
        if self.active.is_empty() { return Ok(()); }
        loop {
            match self.ring.submit_and_wait(usize::from(wait && !progressed)) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        self.reap()?;
        Ok(())
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.queued.clear();
        // A failed submission may have allocated an operation without publishing an SQE.
        // Those allocations have no kernel owner and will never produce a CQE to wait for.
        self.active.retain(|_, operation| operation.in_kernel);
        while !self.active.is_empty() {
            if self.pump(true).is_err() {
                // A fatal ring failure must not free memory the kernel might still access.
                // Preserve the allocations and file descriptors rather than risk use-after-free.
                #[allow(clippy::mem_forget, reason = "The kernel may still own these allocations")]
                std::mem::forget(std::mem::take(&mut self.active));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests;
