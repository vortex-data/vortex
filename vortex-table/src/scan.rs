// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scanning a table: a planner over its manifest that opens each file it keeps as an independent
//! child over the file's own bytes.
//!
//! ```text
//! TablePlanner                       no IO: walks the manifest
//! └── Open ─▶ FilePlanner            reads the footer region, unless the manifest inlines it,
//!             │                      prunes the file on its footer statistics,
//!             │                      and prepares the file's scan
//!             └── Open ─▶ compose()  the file's own prune and chunk planners
//! ```
//!
//! Every edge is an [`Open`](PlannerOutput::Open): the table, each file's footer, and each file's
//! segments are separate byte sources, and nothing in the plan says how many run at once. That
//! is the driver's choice.

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::v2::compose;
use vortex_io::VortexReadAt;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSlot;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::request::ReadAtIoService;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::planning::drive::PlanDriver;
use vortex_scan::planning::drive::PlanRoot;
use vortex_scan::planning::drive::ScopedBatch;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::footer::footer_hash;
use crate::footer::parse_footer;
use crate::manifest::Manifest;
use crate::manifest::ManifestEntry;
use crate::storage::TableStorage;

/// A scan of a table: a projection and filter over every file of its manifest.
pub struct TableScan {
    session: VortexSession,
    storage: Arc<dyn TableStorage>,
    manifest: Arc<Manifest>,
    projection: Expression,
    filter: Option<Expression>,
    selections: Option<HashMap<u64, Selection>>,
    cache: Option<Arc<FileCache>>,
}

/// Files a table's scans have opened, kept so later scans skip the footer and reuse each
/// file's scan plans. Keyed by path and footer hash, so a rewritten file is opened afresh.
#[derive(Default)]
pub struct FileCache {
    files: Mutex<HashMap<(String, u64), VortexFile>>,
}

impl FileCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn get(&self, entry: &ManifestEntry) -> Option<VortexFile> {
        self.files
            .lock()
            .get(&(entry.path.clone(), entry.footer_hash))
            .cloned()
    }

    fn insert(&self, entry: &ManifestEntry, file: VortexFile) {
        self.files
            .lock()
            .insert((entry.path.clone(), entry.footer_hash), file);
    }
}

impl TableScan {
    /// A scan of every row and column of the table.
    pub fn new(
        session: VortexSession,
        storage: Arc<dyn TableStorage>,
        manifest: Arc<Manifest>,
    ) -> Self {
        Self {
            session,
            storage,
            manifest,
            projection: root(),
            filter: None,
            selections: None,
            cache: None,
        }
    }

    /// Projects each row with `projection`.
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    /// Keeps the rows `filter` is true for. Files whose footer statistics prove it false for
    /// every row are pruned before their segments are read.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Restricts the scan to the given rows of the given files, by file ordinal, such as the
    /// matches an index found. Files with no selection are not opened at all.
    pub fn with_file_selections(mut self, selections: HashMap<u64, Selection>) -> Self {
        self.selections = Some(selections);
        self
    }

    /// Opens files through `cache`, and keeps the files this scan opens there.
    pub fn with_file_cache(mut self, cache: Arc<FileCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// The scan's root planner, ready for any [`PlanDriver`].
    pub fn plan(self) -> PlanRoot {
        let mut row_offset = 0;
        let files = self
            .manifest
            .entries()
            .iter()
            .enumerate()
            .filter_map(|(ordinal, entry)| {
                let ordinal = ordinal as u64;
                let offset = row_offset;
                row_offset += entry.row_count;
                let selection = match &self.selections {
                    None => Selection::All,
                    Some(selections) => selections.get(&ordinal)?.clone(),
                };
                Some(TableFile {
                    ordinal,
                    entry: entry.clone(),
                    selection,
                    row_offset: offset,
                })
            })
            .collect::<Vec<_>>();
        let total = row_offset;
        let shared = Arc::new(Shared {
            session: self.session,
            storage: self.storage,
            projection: self.projection,
            filter: self.filter,
            cache: self.cache,
        });
        PlanRoot::new(
            WorkScope {
                file_ordinal: 0,
                rows: 0..total,
            },
            Box::new(TablePlanner {
                shared,
                files: files.into_iter(),
            }),
            Arc::new(NoIo),
        )
    }

    /// Runs the scan on `driver`.
    pub fn execute(self, driver: &dyn PlanDriver) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        driver.execute(self.plan())
    }
}

/// What every planner of one table scan reads.
struct Shared {
    session: VortexSession,
    storage: Arc<dyn TableStorage>,
    projection: Expression,
    filter: Option<Expression>,
    cache: Option<Arc<FileCache>>,
}

/// One file the scan opens.
struct TableFile {
    ordinal: u64,
    entry: ManifestEntry,
    selection: Selection,
    /// The table row of the file's first row.
    row_offset: u64,
}

/// Opens each file of the scan, one per compute, as an independent child over its own bytes.
struct TablePlanner {
    shared: Arc<Shared>,
    files: std::vec::IntoIter<TableFile>,
}

impl IoConsumer for TablePlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for TablePlanner {
    fn state(&self) -> State {
        if self.files.len() == 0 {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(file) = self.files.next() else {
            return Ok(PlannerOutput::Done);
        };
        let scope = WorkScope {
            file_ordinal: file.ordinal,
            rows: 0..file.entry.row_count,
        };
        let cached = self
            .shared
            .cache
            .as_ref()
            .and_then(|cache| cache.get(&file.entry));
        let (source, io): (FileSource, Arc<dyn IoService>) = match cached {
            Some(opened) => (FileSource::Opened(Box::new(opened)), Arc::new(NoIo)),
            None => {
                let read = self.shared.storage.open(&file.entry.path)?;
                let io = Arc::new(ReadAtIoService::new(Arc::clone(&read)));
                (FileSource::Read(read), io)
            }
        };
        let planner = FilePlanner {
            shared: Arc::clone(&self.shared),
            file,
            source: Some(source),
            slot: IoSlot::default(),
            done: false,
        };
        Ok(PlannerOutput::Open(scope, Box::new(planner), io))
    }
}

/// Opens one file: takes its footer from the manifest or reads it, prunes the file on its
/// statistics, and hands the file's own scan planners to the driver.
struct FilePlanner {
    shared: Arc<Shared>,
    file: TableFile,
    source: Option<FileSource>,
    slot: IoSlot,
    done: bool,
}

/// How a file planner gets its file: already opened by an earlier scan, or from its bytes.
enum FileSource {
    Opened(Box<VortexFile>),
    Read(Arc<dyn VortexReadAt>),
}

impl FilePlanner {
    fn open(
        &mut self,
        read: Arc<dyn VortexReadAt>,
        region: vortex_buffer::ByteBuffer,
    ) -> VortexResult<PlannerOutput> {
        let entry = &self.file.entry;
        vortex_ensure_eq!(
            footer_hash(&region),
            entry.footer_hash,
            "{} changed since the manifest was written",
            entry.path
        );
        let session = &self.shared.session;
        let footer = parse_footer(region, entry.byte_size, session)?;
        // Given its footer and size, and no segment cache or metadata to load, opening a file
        // reads nothing, so the future is ready at once.
        let file = futures::executor::block_on(
            session
                .open_options()
                .with_footer(footer)
                .with_file_size(entry.byte_size)
                .open(read),
        )?;
        let file = match &self.shared.cache {
            Some(cache) => {
                let file = file.with_caching();
                cache.insert(entry, file.clone());
                file
            }
            None => file,
        };
        self.plan(file)
    }

    /// Prunes `file` on its statistics, or prepares its scan and hands over its planners.
    fn plan(&mut self, file: VortexFile) -> VortexResult<PlannerOutput> {
        self.done = true;
        let session = &self.shared.session;
        if let Some(filter) = &self.shared.filter
            && file.can_prune(filter)?
        {
            return Ok(PlannerOutput::Done);
        }
        let dtype = file.dtype().clone();
        let scan = ScanBuilder::new(session.clone(), file.scan_file())
            .with_projection(self.shared.projection.bind(&dtype)?)
            .with_some_filter(
                self.shared
                    .filter
                    .as_ref()
                    .map(|filter| filter.bind(&dtype))
                    .transpose()?,
            )
            .with_selection(self.file.selection.clone())
            .with_row_offset(self.file.row_offset)
            .with_file_ordinal(self.file.ordinal)
            .prepare()?;
        let scan = Arc::new(scan);
        let scope = WorkScope {
            file_ordinal: self.file.ordinal,
            rows: scan.rows().clone(),
        };
        Ok(PlannerOutput::Open(scope, compose(&scan), scan.io()))
    }
}

impl IoConsumer for FilePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl Planner for FilePlanner {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else if self.slot.is_waiting() {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let read = match self.source.take() {
            Some(FileSource::Opened(file)) => return self.plan(*file),
            Some(FileSource::Read(read)) => read,
            None => vortex_bail!("the file planner ran after it finished"),
        };
        if let Some(region) = self.file.entry.footer.clone() {
            return self.open(read, region);
        }
        let result = self.slot.take();
        if result.is_none() {
            // Keep the source for the delivery.
            self.source = Some(FileSource::Read(Arc::clone(&read)));
        }
        match result {
            Some(IoResult::Bytes(region)) => self.open(read, region.try_into_host_sync()?),
            Some(IoResult::Size(_)) => vortex_bail!("a footer read was answered with a size"),
            None => {
                let entry = &self.file.entry;
                let len = usize::try_from(entry.footer_len)?;
                Ok(PlannerOutput::NeedsIO(
                    self.slot.issue(IoTarget::range(entry.footer_offset, len)),
                ))
            }
        }
    }
}

/// The IO service of a planner that reads nothing.
struct NoIo;

impl IoService for NoIo {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(NoIo)
    }
}

impl IoSource for NoIo {
    fn submit(&self, _owner: IoOwnerId, _batch: IoBatch) -> VortexResult<()> {
        vortex_bail!("the table planner reads nothing")
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn poll_completion(&self, _cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        Poll::Ready(Err(vortex_err!("the table planner reads nothing")))
    }

    fn wait(&self) -> VortexResult<Completion> {
        Err(vortex_err!("the table planner reads nothing"))
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {}
}
