// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;

use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use vortex_error::VortexResult;

use crate::LayoutReader;
use crate::plan::PlanRef;
use crate::scan::v2::ScanFile;
use crate::scan::v2::lower::lower;
use crate::scan::v2::lower::lower_with_zones;

/// Lowered plans shared by active scans over one layout reader.
///
/// Dictionary values and zone statistics may be retained while these scans run, as in V1.
/// The registry holds only weak references, so it never extends their lifetime.
pub(super) struct SharedFile {
    pub(super) file: ScanFile,
    /// The file's layout lowered to a plan.
    pub(super) root: PlanRef,
    /// The file's layout lowered with its zone statistics, for pruning.
    zones: OnceCell<PlanRef>,
    _reader: Arc<dyn LayoutReader>,
}

impl SharedFile {
    pub(super) fn zones(&self) -> VortexResult<PlanRef> {
        self.zones
            .get_or_try_init(|| lower_with_zones(&self.file.layout))
            .cloned()
    }
}

/// A reader and a weak reference to its active scans' shared file.
type Entry = (Weak<dyn LayoutReader>, Arc<Mutex<Weak<SharedFile>>>);

static FILES: LazyLock<Mutex<Vec<Entry>>> = LazyLock::new(Default::default);

/// The shared file of `reader`, created over `file` for its first scan.
pub(super) fn shared_file(
    reader: &Arc<dyn LayoutReader>,
    file: ScanFile,
) -> VortexResult<Arc<SharedFile>> {
    let cell = {
        let mut files = FILES.lock();
        files.retain(|(reader, _)| reader.strong_count() > 0);
        if let Some((_, cell)) = files
            .iter()
            .find(|(known, _)| ptr::addr_eq(known.as_ptr(), Arc::as_ptr(reader)))
        {
            Arc::clone(cell)
        } else {
            let cell = Arc::default();
            files.push((Arc::downgrade(reader), Arc::clone(&cell)));
            cell
        }
    };
    // Partitions of one file share initialization, while different files can lower in parallel.
    let mut known = cell.lock();
    if let Some(shared) = known.upgrade() {
        return Ok(shared);
    }
    let shared = Arc::new(SharedFile {
        root: lower(&file.layout)?,
        zones: OnceCell::default(),
        _reader: Arc::clone(reader),
        file,
    });
    *known = Arc::downgrade(&shared);
    Ok(shared)
}
