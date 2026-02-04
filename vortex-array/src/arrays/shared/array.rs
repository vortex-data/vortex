// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::stats::ArrayStats;
use vortex_dtype::DType;

#[derive(Debug, Clone)]
pub struct SharedArray {
    pub(super) state: Arc<RwLock<SharedState>>,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

#[derive(Debug, Clone)]
pub(super) enum SharedState {
    Source(ArrayRef),
    Cached(Canonical),
}

impl SharedArray {
    pub fn new(source: ArrayRef) -> Self {
        Self {
            dtype: source.dtype().clone(),
            state: Arc::new(RwLock::new(SharedState::Source(source))),
            stats: ArrayStats::default(),
        }
    }

    pub fn cached(&self) -> Option<Canonical> {
        match &*self.state.read() {
            SharedState::Cached(canonical) => Some(canonical.clone()),
            SharedState::Source(_) => None,
        }
    }

    pub fn cache_or_return(&self, canonical: Canonical) -> Canonical {
        let mut state = self.state.write();
        match &*state {
            SharedState::Cached(existing) => existing.clone(),
            SharedState::Source(_) => {
                *state = SharedState::Cached(canonical.clone());
                canonical
            }
        }
    }

    pub fn source_if_any(&self) -> Option<ArrayRef> {
        match &*self.state.read() {
            SharedState::Source(source) => Some(source.clone()),
            SharedState::Cached(_) => None,
        }
    }

    pub(super) fn canonicalize(&self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(existing) = self.cached() {
            return Ok(existing);
        }
        let source = match self.source_if_any() {
            Some(source) => source,
            None => {
                return Ok(
                    self.cached()
                        .vortex_expect("cache present when no source"),
                )
            }
        };
        let canonical = source.execute::<Canonical>(ctx)?;
        Ok(self.cache_or_return(canonical))
    }

    pub(super) fn current_array_ref(&self) -> ArrayRef {
        match &*self.state.read() {
            SharedState::Source(source) => source.clone(),
            SharedState::Cached(canonical) => canonical.clone().into_array(),
        }
    }

    pub(super) fn set_source(&mut self, source: ArrayRef) {
        self.dtype = source.dtype().clone();
        *self.state.write() = SharedState::Source(source);
    }

    pub(super) fn visit_children(&self, visitor: &mut dyn crate::ArrayChildVisitor) {
        let child = self.current_array_ref();
        visitor.visit_child("source", &child);
    }
}
