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
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;
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
            segment_sources: Arc::new(Mutex::new(SegmentSourceRegistry::default())),
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

/// Scheduler-local identity for a registered segment source.
///
/// The identity is scoped to one [`ScanTicket`]. A shared scheduler may later associate this with a
/// stable cross-scan source key for cache reuse or metrics, but correctness must not depend on two
/// tickets allocating the same value for the same object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentSourceId(u64);

impl SegmentSourceId {
    /// Return the integer value of this scheduler-local source id.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Metadata attached to a registered segment source.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SegmentSourceMeta {
    /// Optional human-readable label used for diagnostics and future metrics.
    pub label: Option<String>,
}

#[derive(Default)]
struct SegmentSourceRegistry {
    next_id: u64,
    sources: HashMap<SegmentSourceId, SegmentSourceEntry>,
}

struct SegmentSourceEntry {
    source: Arc<dyn Any + Send + Sync>,
    meta: SegmentSourceMeta,
}

/// A logical scan registered with a scheduler.
#[derive(Clone)]
pub struct ScanTicket {
    id: u64,
    cancelled: Arc<AtomicBool>,
    per_scan_slots: Option<Arc<Semaphore>>,
    per_scan_slot_limit: Option<usize>,
    segment_sources: Arc<Mutex<SegmentSourceRegistry>>,
}

impl fmt::Debug for ScanTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanTicket")
            .field("id", &self.id)
            .field("cancelled", &self.is_cancelled())
            .field("per_scan_slot_limit", &self.per_scan_slot_limit)
            .field("segment_source_count", &self.segment_source_count())
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

    /// Register a segment source and return its scan-local source id.
    pub fn register_segment_source<S>(
        &self,
        source: Arc<S>,
        meta: SegmentSourceMeta,
    ) -> SegmentSourceId
    where
        S: Any + Send + Sync,
    {
        let source: Arc<dyn Any + Send + Sync> = source;
        self.register_erased_segment_source(source, meta)
    }

    /// Register an already-erased segment source and return its scan-local source id.
    pub fn register_erased_segment_source(
        &self,
        source: Arc<dyn Any + Send + Sync>,
        meta: SegmentSourceMeta,
    ) -> SegmentSourceId {
        let mut registry = self.segment_sources.lock();
        let id = SegmentSourceId(registry.next_id);
        registry.next_id = registry.next_id.saturating_add(1);
        registry
            .sources
            .insert(id, SegmentSourceEntry { source, meta });
        id
    }

    /// Return the metadata for a registered segment source.
    pub fn segment_source_meta(&self, id: SegmentSourceId) -> Option<SegmentSourceMeta> {
        let registry = self.segment_sources.lock();
        registry.sources.get(&id).map(|entry| entry.meta.clone())
    }

    /// Return a registered segment source downcast to the requested concrete type.
    ///
    /// This is intentionally typed at the call site: the scheduler stores sources opaquely, while
    /// the scan runtime decides which concrete source trait or adapter it expects.
    pub fn segment_source<S>(&self, id: SegmentSourceId) -> Option<Arc<S>>
    where
        S: Any + Send + Sync,
    {
        let source = {
            let registry = self.segment_sources.lock();
            Arc::clone(&registry.sources.get(&id)?.source)
        };
        source.downcast::<S>().ok()
    }

    fn segment_source_count(&self) -> usize {
        let registry = self.segment_sources.lock();
        registry.sources.len()
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
    fn with_scan_scheduler(self, scheduler: Arc<ScanScheduler>) -> Self {
        self.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::Shared(scheduler));
        self
    }

    /// Configure this session to create one scheduler per logical scan.
    fn with_new_scan_scheduler_per_scan(self, config: ScanSchedulerConfig) -> Self {
        self.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::PerScan(config));
        self
    }

    /// Configure this session to run scans without scheduler limits.
    fn with_unbounded_scan_scheduler(self) -> Self {
        self.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::Unbounded);
        self
    }
}

impl<S: SessionExt> ScanSchedulerSessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;

    struct TestSegmentSource {
        label: &'static str,
    }

    #[test]
    fn segment_source_registration_is_scan_local() -> VortexResult<()> {
        let scheduler = ScanScheduler::unbounded();
        let scan_a = scheduler.register_scan(ScanMeta::default());
        let scan_b = scheduler.register_scan(ScanMeta::default());

        let source_a = Arc::new(TestSegmentSource { label: "a" });
        let source_b = Arc::new(TestSegmentSource { label: "b" });

        let id_a0 = scan_a.register_segment_source(
            source_a,
            SegmentSourceMeta {
                label: Some("source-a".to_string()),
            },
        );
        let id_a1 = scan_a.register_segment_source(
            Arc::new(TestSegmentSource { label: "a1" }),
            SegmentSourceMeta::default(),
        );
        let id_b0 = scan_b.register_segment_source(source_b, SegmentSourceMeta::default());

        assert_eq!(id_a0.get(), 0);
        assert_eq!(id_a1.get(), 1);
        assert_eq!(id_b0.get(), 0);

        let meta = scan_a
            .segment_source_meta(id_a0)
            .ok_or_else(|| vortex_err!("missing segment source metadata"))?;
        assert_eq!(meta.label.as_deref(), Some("source-a"));

        let source = scan_a
            .segment_source::<TestSegmentSource>(id_a0)
            .ok_or_else(|| vortex_err!("missing registered segment source"))?;
        assert_eq!(source.label, "a");
        assert!(scan_b.segment_source::<TestSegmentSource>(id_a1).is_none());

        Ok(())
    }
}
