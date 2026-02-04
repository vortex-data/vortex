// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::OnceLock;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::stats::ArrayStats;

#[derive(Debug)]
pub struct SharedArray {
    pub(super) source: ArrayRef,
    pub(super) cache: Arc<OnceLock<Canonical>>,
    pub(super) stats: ArrayStats,
}

impl Clone for SharedArray {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            cache: Arc::clone(&self.cache),
            stats: self.stats.clone(),
        }
    }
}

impl SharedArray {
    pub fn new(source: ArrayRef) -> Self {
        Self {
            source,
            cache: Arc::new(OnceLock::new()),
            stats: ArrayStats::default(),
        }
    }

    pub fn source(&self) -> &ArrayRef {
        &self.source
    }

    pub fn cached(&self) -> Option<Canonical> {
        self.cache.get().cloned()
    }

    pub fn cache_or_return(&self, canonical: Canonical) -> Canonical {
        if let Some(existing) = self.cache.get() {
            return existing.clone();
        }
        drop(self.cache.set(canonical.clone()));
        canonical
    }

    pub(super) fn canonicalize(&self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(existing) = self.cache.get() {
            return Ok(existing.clone());
        }
        let canonical = self.source.clone().execute::<Canonical>(ctx)?;
        drop(self.cache.set(canonical.clone()));
        Ok(canonical)
    }
}
