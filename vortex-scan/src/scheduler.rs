// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Coarse-grained resource scheduling for scans.
//!
//! The scheduler deliberately starts with one primitive: a slot permit. The ScanPlan runtime
//! uses one slot per in-flight morsel, which is enough to preserve the existing scan concurrency
//! model while giving integrations a shared object they can use to bound concurrent work across
//! scans.

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_lock::Semaphore;
use async_lock::SemaphoreGuardArc;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

const DEFAULT_MORSEL_CONCURRENCY_FACTOR: usize = 4;

/// Configuration for a [`ScanScheduler`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanSchedulerConfig {
    global_slots: Option<usize>,
    per_scan_slots: Option<usize>,
    morsel_plan_window: Option<usize>,
    morsel_launch_window: Option<usize>,
}

impl ScanSchedulerConfig {
    /// Create an unbounded scheduler configuration.
    pub fn unbounded() -> Self {
        Self {
            global_slots: None,
            per_scan_slots: None,
            morsel_plan_window: None,
            morsel_launch_window: None,
        }
    }

    /// Create a scheduler configuration with the same morsel-slot limit globally and per scan.
    ///
    /// Morsel execution remains bounded by `slots`, but planning is unbounded by default so
    /// segment futures can be registered ahead of execution.
    pub fn morsel_slots(slots: usize) -> Self {
        let slots = slots.max(1);
        Self {
            global_slots: Some(slots),
            per_scan_slots: Some(slots),
            morsel_plan_window: None,
            morsel_launch_window: Some(slots),
        }
    }

    /// Return a copy with the maximum number of morsels allowed to be planned ahead per scan.
    ///
    /// `None` means the scan may plan all morsels ahead of execution.
    pub fn with_morsel_plan_window(mut self, window: Option<usize>) -> Self {
        self.morsel_plan_window = window.map(|window| window.max(1));
        self
    }

    /// Return a copy with the maximum number of morsels allowed to run concurrently per scan.
    pub fn with_morsel_launch_window(mut self, window: Option<usize>) -> Self {
        self.morsel_launch_window = window.map(|window| window.max(1));
        self
    }

    /// Create a scheduler configuration matching the current unordered scan concurrency factor.
    pub fn default_morsel_slots() -> Self {
        Self::morsel_slots(default_morsel_slots())
    }

    /// Configuration used by the DuckDB integration by default.
    pub fn duckdb_default() -> Self {
        Self::default_morsel_slots()
    }

    /// Returns the configured global slot limit.
    pub fn global_slots(&self) -> Option<usize> {
        self.global_slots
    }

    /// Returns the configured per-scan slot limit.
    pub fn per_scan_slots(&self) -> Option<usize> {
        self.per_scan_slots
    }

    /// Returns the configured per-scan morsel planning window.
    ///
    /// `None` means planning is unbounded.
    pub fn morsel_plan_window(&self) -> Option<usize> {
        self.morsel_plan_window
    }

    /// Returns the configured per-scan morsel launch window.
    pub fn morsel_launch_window(&self) -> Option<usize> {
        self.morsel_launch_window
    }
}

impl Default for ScanSchedulerConfig {
    fn default() -> Self {
        Self::default_morsel_slots()
    }
}

/// Returns the default number of morsel slots for unordered scans.
pub fn default_morsel_slots() -> usize {
    get_available_parallelism()
        .unwrap_or(1)
        .saturating_mul(DEFAULT_MORSEL_CONCURRENCY_FACTOR)
        .max(1)
}

/// Shared scheduler that admits scan work using coarse slot permits.
pub struct ScanScheduler {
    config: ScanSchedulerConfig,
    global_slots: Option<Arc<Semaphore>>,
    next_scan_id: AtomicU64,
}

impl fmt::Debug for ScanScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ScanScheduler {
    /// Create a scheduler from a configuration.
    pub fn new(config: ScanSchedulerConfig) -> Self {
        let global_slots = config
            .global_slots
            .map(|slots| Arc::new(Semaphore::new(slots)));
        Self {
            config,
            global_slots,
            next_scan_id: AtomicU64::new(0),
        }
    }

    /// Create an unbounded scheduler.
    pub fn unbounded() -> Self {
        Self::new(ScanSchedulerConfig::unbounded())
    }

    /// Return this scheduler's configuration.
    pub fn config(&self) -> &ScanSchedulerConfig {
        &self.config
    }

    /// Register a logical scan and return a ticket used for future permit acquisition.
    pub fn register_scan(&self, _meta: ScanMeta) -> ScanTicket {
        let id = self.next_scan_id.fetch_add(1, Ordering::Relaxed);
        ScanTicket {
            id,
            cancelled: Arc::new(AtomicBool::new(false)),
            per_scan_slots: self
                .config
                .per_scan_slots
                .map(|slots| Arc::new(Semaphore::new(slots))),
            per_scan_slot_limit: self.config.per_scan_slots,
        }
    }

    /// Acquire permits for one unit of scan work.
    pub async fn acquire(
        &self,
        ticket: &ScanTicket,
        request: WorkRequest,
    ) -> VortexResult<WorkPermit> {
        if ticket.is_cancelled() {
            vortex_bail!("scan {} was cancelled", ticket.id());
        }

        let slots = usize::try_from(request.slots)
            .map_err(|_| vortex_error::vortex_err!("scan work slot count exceeds usize"))?;
        if slots == 0 {
            vortex_bail!("scan work must request at least one scheduler slot");
        }
        if slots != 1 {
            vortex_bail!("the MVP scan scheduler only supports one slot per work request");
        }
        if let Some(limit) = ticket.per_scan_slot_limit
            && slots > limit
        {
            vortex_bail!(
                "scan work requested {} slots, exceeding per-scan limit {}",
                slots,
                limit
            );
        }
        if let Some(limit) = self.config.global_slots
            && slots > limit
        {
            vortex_bail!(
                "scan work requested {} slots, exceeding global limit {}",
                slots,
                limit
            );
        }

        let mut guards = Vec::with_capacity(slots.saturating_mul(2));
        if let Some(per_scan_slots) = &ticket.per_scan_slots {
            for _ in 0..slots {
                guards.push(per_scan_slots.acquire_arc().await);
            }
        }
        if let Some(global_slots) = &self.global_slots {
            for _ in 0..slots {
                guards.push(global_slots.acquire_arc().await);
            }
        }

        if ticket.is_cancelled() {
            vortex_bail!("scan {} was cancelled", ticket.id());
        }

        Ok(WorkPermit { _guards: guards })
    }
}

/// Scheduler ownership strategy.
#[derive(Clone, Debug, Default)]
pub enum ScanSchedulerProvider {
    /// Use one scheduler for every scan that shares this provider.
    Shared(Arc<ScanScheduler>),
    /// Create a new scheduler whenever a logical scan starts.
    PerScan(ScanSchedulerConfig),
    /// Do not bound scan work through scheduler permits.
    #[default]
    Unbounded,
}

impl ScanSchedulerProvider {
    /// Resolve the scheduler used for a logical scan.
    pub fn scheduler_for_scan(&self, _meta: &ScanMeta) -> Arc<ScanScheduler> {
        match self {
            Self::Shared(scheduler) => Arc::clone(scheduler),
            Self::PerScan(config) => Arc::new(ScanScheduler::new(config.clone())),
            Self::Unbounded => Arc::new(ScanScheduler::unbounded()),
        }
    }
}

/// Metadata for a logical scan registered with a [`ScanScheduler`].
#[derive(Clone, Debug, Default)]
pub struct ScanMeta {
    /// Optional label used for diagnostics and future metrics.
    pub label: Option<String>,
}

/// A logical scan registered with a scheduler.
#[derive(Clone)]
pub struct ScanTicket {
    id: u64,
    cancelled: Arc<AtomicBool>,
    per_scan_slots: Option<Arc<Semaphore>>,
    per_scan_slot_limit: Option<usize>,
}

impl fmt::Debug for ScanTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanTicket")
            .field("id", &self.id)
            .field("cancelled", &self.is_cancelled())
            .field("per_scan_slot_limit", &self.per_scan_slot_limit)
            .finish_non_exhaustive()
    }
}

impl ScanTicket {
    /// Return this ticket's scheduler-local scan id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Cancel the scan.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Return whether the scan has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// A request to acquire scheduler slots for one scan work item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkRequest {
    /// The class of scan work requesting admission.
    pub class: ScanWorkClass,
    /// Number of slots requested.
    ///
    /// The MVP scheduler only accepts `1`.
    pub slots: u32,
}

impl WorkRequest {
    /// Create a request for one morsel execution slot.
    pub fn morsel() -> Self {
        Self {
            class: ScanWorkClass::Morsel,
            slots: 1,
        }
    }
}

/// Coarse scan work classes understood by the MVP scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanWorkClass {
    /// File-open work.
    FileOpen,
    /// Morsel execution work.
    Morsel,
    /// Output conversion work.
    OutputConversion,
}

/// RAII scheduler permit.
///
/// Dropping this value releases every scheduler slot acquired for a work item.
pub struct WorkPermit {
    _guards: Vec<SemaphoreGuardArc>,
}

impl fmt::Debug for WorkPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkPermit")
            .field("slot_count", &self._guards.len())
            .finish()
    }
}

/// Session state for scan scheduler configuration.
#[derive(Clone, Debug)]
pub struct ScanSchedulerSession {
    provider: Arc<ScanSchedulerProvider>,
}

impl ScanSchedulerSession {
    /// Create a session variable from a scheduler provider.
    pub fn new(provider: Arc<ScanSchedulerProvider>) -> Self {
        Self { provider }
    }

    /// Return the configured scheduler provider.
    pub fn provider(&self) -> Arc<ScanSchedulerProvider> {
        Arc::clone(&self.provider)
    }
}

impl Default for ScanSchedulerSession {
    fn default() -> Self {
        Self {
            provider: Arc::new(ScanSchedulerProvider::default()),
        }
    }
}

impl SessionVar for ScanSchedulerSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for configuring scan scheduler session state.
pub trait ScanSchedulerSessionExt: SessionExt {
    /// Return the configured scan scheduler provider.
    fn scan_scheduler_provider(&self) -> Arc<ScanSchedulerProvider> {
        self.get::<ScanSchedulerSession>().provider()
    }

    /// Configure this session to share one scheduler across scans.
    fn with_scan_scheduler(self, scheduler: Arc<ScanScheduler>) -> VortexSession {
        let mut builder = self.session().to_builder();
        builder.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::Shared(scheduler));
        builder.build()
    }

    /// Configure this session to create one scheduler per logical scan.
    fn with_new_scan_scheduler_per_scan(self, config: ScanSchedulerConfig) -> VortexSession {
        let mut builder = self.session().to_builder();
        builder.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::PerScan(config));
        builder.build()
    }

    /// Configure this session to run scans without scheduler limits.
    fn with_unbounded_scan_scheduler(self) -> VortexSession {
        let mut builder = self.session().to_builder();
        builder.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::Unbounded);
        builder.build()
    }
}

impl<S: SessionExt> ScanSchedulerSessionExt for S {}
