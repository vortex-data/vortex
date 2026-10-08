// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;

use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LayoutReader;
use crate::plan::PlanRef;
use crate::scan::v2::ScanFile;
use crate::scan::v2::lower::lower;
use crate::scan::v2::lower::lower_with_zones;
use crate::scan::v2::repeated_scan::PreparedPlans;

/// A file's lowered plans, shared by every scan over the same layout reader.
///
/// The plans hold what executions of them learn about the file, such as dictionary values and
/// which zones a filter prunes, so scans that share them read and decode those once. Engines share
/// one layout reader between the scans of a file, as DataFusion does across partitions, and the
/// default scan shares the same state through that reader.
pub(super) struct SharedFile {
    pub(super) file: ScanFile,
    /// The file's layout lowered to a plan.
    pub(super) root: PlanRef,
    /// The file's layout lowered with its zone statistics, for pruning.
    pub(super) zones: PlanRef,
    plans: Mutex<HashMap<ScanExpressions, Arc<OnceCell<Arc<PreparedPlans>>>>>,
}

type ScanExpressions = (BoundExpression, Option<BoundExpression>);

impl SharedFile {
    /// Partitions of one reader use the same expression plans and full-file chunk boundaries.
    /// Row ranges, masks, decoding, and output mapping remain local to each scan.
    pub(super) fn prepare_plans(
        &self,
        projection: &BoundExpression,
        filter: &Option<BoundExpression>,
        session: &VortexSession,
    ) -> VortexResult<Arc<PreparedPlans>> {
        let cell = Arc::clone(
            self.plans
                .lock()
                .entry((projection.clone(), filter.clone()))
                .or_default(),
        );
        cell.get_or_try_init(|| PreparedPlans::new(self, projection, filter, session).map(Arc::new))
            .cloned()
    }
}

/// A reader and the shared file of its scans.
type Entry = (Weak<dyn LayoutReader>, Arc<SharedFile>);

/// Every live reader's shared file. An entry lives as long as its reader.
static FILES: LazyLock<Mutex<Vec<Entry>>> = LazyLock::new(Default::default);

/// The shared file of `reader`, created over `file` for its first scan.
pub(super) fn shared_file(
    reader: &Arc<dyn LayoutReader>,
    file: ScanFile,
) -> VortexResult<Arc<SharedFile>> {
    let mut files = FILES.lock();
    files.retain(|(reader, _)| reader.strong_count() > 0);
    if let Some((_, shared)) = files
        .iter()
        .find(|(known, _)| ptr::addr_eq(known.as_ptr(), Arc::as_ptr(reader)))
    {
        return Ok(Arc::clone(shared));
    }
    let shared = Arc::new(SharedFile {
        root: lower(&file.layout)?,
        zones: lower_with_zones(&file.layout)?,
        plans: Mutex::default(),
        file,
    });
    files.push((Arc::downgrade(reader), Arc::clone(&shared)));
    Ok(shared)
}
