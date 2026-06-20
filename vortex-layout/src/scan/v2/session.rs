// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-level default for which scan implementation `VortexScanExec` expands to.

use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_session::SessionExt;
use vortex_session::SessionVar;
/// Session variable holding the scan implementation default.
#[derive(Debug)]
pub struct ScanV2Session {
    /// Whether `VortexScanExec` expands through scan2 when the node does
    /// not choose explicitly (see `VortexScanExec::with_scan2`).
    default_enabled: AtomicBool,
}

impl Clone for ScanV2Session {
    fn clone(&self) -> Self {
        Self {
            default_enabled: AtomicBool::new(self.default_enabled()),
        }
    }
}

impl ScanV2Session {
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
        Self {
            default_enabled: AtomicBool::new(false),
        }
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

/// Session accessor for the scan2 implementation switch.
pub trait ScanV2SessionExt: SessionExt {
    /// The scan2 session variable.
    fn scan_v2(&self) -> &ScanV2Session {
        self.get::<ScanV2Session>()
    }
}

impl<S: SessionExt> ScanV2SessionExt for S {}
