// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::stats::ArrayStats;

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

    pub fn as_source(&self) -> ArrayRef {
        let SharedState::Source(source) = &*self.state.read() else {
            vortex_panic!("already cached");
        };
        source.clone()
    }

    pub(super) fn canonicalize(&self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(existing) = self.cached() {
            return Ok(existing);
        }
        let canonical = self.as_source().execute::<Canonical>(ctx)?;
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
}
