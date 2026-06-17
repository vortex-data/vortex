// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry for layout scan2 rules, plus the session-level
//! default for which scan implementation `VortexScanExec` expands to.

use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::LayoutEncodingId;
use crate::scan::v2::layouts::chunked::ChunkedScanRule;
use crate::scan::v2::layouts::dict::DictScanRule;
use crate::scan::v2::layouts::flat::FlatScanRule;
use crate::scan::v2::layouts::struct_::StructScanRule;
use crate::scan::v2::layouts::zoned::ZonedScanRule;
use crate::scan::v2::node::LayoutScanRule;
use crate::scan::v2::node::ScanRuleRef;

/// The registry mapping layout encodings to scan2 rules.
pub type ScanRuleRegistry = Registry<ScanRuleRef>;

/// Session variable holding the engine's layout scan2 rules, keyed by
/// [`LayoutEncodingId`], and the session default for the scan
/// implementation swap. The default registers the built-in rules (flat,
/// chunked, struct, dict, zoned); third-party layout crates register
/// their own the same way.
#[derive(Debug)]
pub struct ScanV2Session {
    registry: ScanRuleRegistry,
    /// Whether `VortexScanExec` expands through scan2 when the node does
    /// not choose explicitly (see `VortexScanExec::with_scan2`).
    default_enabled: AtomicBool,
}

impl ScanV2Session {
    /// Register a scan2 rule for the layout encoding it names.
    pub fn register<R: LayoutScanRule>(&self, rule: R) {
        self.registry.register(
            LayoutScanRule::id(&rule),
            std::sync::Arc::new(rule) as ScanRuleRef,
        );
    }

    /// Find the rule registered for a layout encoding.
    pub fn find(&self, id: &LayoutEncodingId) -> Option<ScanRuleRef> {
        self.registry.find(id)
    }

    /// The underlying registry.
    pub fn registry(&self) -> &ScanRuleRegistry {
        &self.registry
    }

    /// Whether scans expand through scan2 by default in this session.
    pub fn default_enabled(&self) -> bool {
        self.default_enabled.load(Ordering::Relaxed)
    }

    /// Set the session default: scans that do not choose explicitly
    /// expand through scan2 when `enabled`.
    pub fn set_default_enabled(&self, enabled: bool) {
        self.default_enabled.store(enabled, Ordering::Relaxed);
    }
}

impl Default for ScanV2Session {
    fn default() -> Self {
        let session = Self {
            registry: ScanRuleRegistry::default(),
            default_enabled: AtomicBool::new(false),
        };
        session.register(FlatScanRule);
        session.register(ChunkedScanRule);
        session.register(StructScanRule);
        session.register(DictScanRule);
        session.register(ZonedScanRule);
        session
    }
}

impl SessionVar for ScanV2Session {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Session accessor for the engine's scan2 rules.
pub trait ScanV2SessionExt: SessionExt {
    /// The layout scan2 rules registered with this session.
    fn scan_v2_rules(&self) -> vortex_session::Ref<'_, ScanV2Session> {
        self.get::<ScanV2Session>()
    }
}

impl<S: SessionExt> ScanV2SessionExt for S {}
