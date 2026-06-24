// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared scan scheduling configuration.
//!
//! The V2 scan runtime currently enforces one scheduler knob: an active logical read-byte budget
//! per partition stream. Scheduler instances still provide shared configuration,
//! but they do not expose a separate morsel-slot permit API.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

const DEFAULT_READ_BYTE_BUDGET: u64 = 256 * 1024 * 1024;

/// Configuration for a [`ScanScheduler`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanSchedulerConfig {
    read_byte_budget: Option<u64>,
}

impl ScanSchedulerConfig {
    /// Create an unbounded scheduler configuration.
    pub fn unbounded() -> Self {
        Self {
            read_byte_budget: None,
        }
    }

    /// Create a scheduler configuration with the default active read-byte budget.
    pub fn default_read_byte_budget() -> Self {
        Self {
            read_byte_budget: Some(DEFAULT_READ_BYTE_BUDGET),
        }
    }

    /// Return a copy with the maximum number of logical read bytes allowed in flight per partition.
    ///
    /// `None` means scan task launch is not capped by bytes.
    pub fn with_read_byte_budget(mut self, bytes: Option<u64>) -> Self {
        self.read_byte_budget = bytes.map(|bytes| bytes.max(1));
        self
    }

    /// Configuration used by the DuckDB integration by default.
    pub fn duckdb_default() -> Self {
        Self::default_read_byte_budget()
    }

    /// Returns the configured per-partition active logical read-byte budget.
    pub fn read_byte_budget(&self) -> Option<u64> {
        self.read_byte_budget
    }
}

impl Default for ScanSchedulerConfig {
    fn default() -> Self {
        Self::default_read_byte_budget()
    }
}

/// Shared scheduler configuration for scan work.
pub struct ScanScheduler {
    config: ScanSchedulerConfig,
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
        Self { config }
    }

    /// Create an unbounded scheduler.
    pub fn unbounded() -> Self {
        Self::new(ScanSchedulerConfig::unbounded())
    }

    /// Return this scheduler's configuration.
    pub fn config(&self) -> &ScanSchedulerConfig {
        &self.config
    }
}

/// Scheduler ownership strategy.
#[derive(Clone, Debug, Default)]
pub enum ScanSchedulerProvider {
    /// Use one scheduler for every scan that shares this provider.
    Shared(Arc<ScanScheduler>),
    /// Create a new scheduler whenever a logical scan starts.
    PerScan(ScanSchedulerConfig),
    /// Do not bound scan work through scheduler configuration.
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

/// Metadata for resolving a logical scan scheduler.
#[derive(Clone, Debug, Default)]
pub struct ScanMeta {
    /// Optional label used for diagnostics and future metrics.
    pub label: Option<String>,
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

    /// Configure this session to run scans without scheduler read-byte limits.
    fn with_unbounded_scan_scheduler(self) -> VortexSession {
        let mut builder = self.session().to_builder();
        builder.get_mut::<ScanSchedulerSession>().provider =
            Arc::new(ScanSchedulerProvider::Unbounded);
        builder.build()
    }
}

impl<S: SessionExt> ScanSchedulerSessionExt for S {}
