// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Verified eviction for cold local-file benchmarks on Linux.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Duration;

/// Evicts a local file from the OS page cache and verifies zero resident pages.
///
/// The caller must first release any mappings and buffers retained by completed scans.
pub fn evict_file(path: &Path) -> anyhow::Result<usize> {
    let file = File::open(path)?;
    let length = usize::try_from(file.metadata()?.len())?;
    if length == 0 {
        return Ok(0);
    }
    // SAFETY: sysconf has no pointer arguments.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    anyhow::ensure!(page_size > 0, "Could not determine page size");
    let pages = length.div_ceil(usize::try_from(page_size)?);
    // SAFETY: mapping a valid file with PROT_NONE does not access or fault in data. We never
    // dereference the mapping; it is used only to query residency and is then unmapped.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_NONE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if mapping == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut residency = vec![0u8; pages];
    // DONTNEED is advisory and does not guarantee immediate eviction. Retry outside the
    // timed region, requiring the same zero-residency condition on every run.
    let evicted = (|| -> anyhow::Result<()> {
        let mut resident = pages;
        for _ in 0..100 {
            // SAFETY: the descriptor remains open; this only advises on its cache pages.
            let error =
                unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            if error != 0 {
                return Err(std::io::Error::from_raw_os_error(error).into());
            }
            // SAFETY: the mapping is page-aligned and alive, and residency has one byte per page.
            let result = unsafe { libc::mincore(mapping, length, residency.as_mut_ptr()) };
            if result != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            resident = residency.iter().filter(|value| **value & 1 != 0).count();
            if resident == 0 {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        anyhow::bail!(
            "{} still has {resident}/{pages} resident pages after eviction",
            path.display()
        )
    })();
    // SAFETY: this is exactly the mapping and length returned above, with no remaining users.
    let unmap = unsafe { libc::munmap(mapping, length) };
    evicted?;
    if unmap != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(pages)
}
