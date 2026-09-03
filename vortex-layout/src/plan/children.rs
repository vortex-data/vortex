// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::PlanRef;

type ChildInitializer = dyn Fn(usize) -> VortexResult<PlanRef> + 'static + Send + Sync;

/// Ordered plan children that may be initialized one slot at a time.
///
/// Eagerly constructed operators store already-filled slots. Layout lowering instead installs an
/// initializer that owns the source layout and lowers each child on first access.
#[derive(Clone)]
pub struct PlanChildren {
    initializer: Option<Arc<ChildInitializer>>,
    cache: Arc<[OnceCell<PlanRef>]>,
}

impl PlanChildren {
    /// Creates lazy child slots backed by `initializer`.
    pub(crate) fn lazy(
        len: usize,
        initializer: impl Fn(usize) -> VortexResult<PlanRef> + 'static + Send + Sync,
    ) -> Self {
        Self {
            initializer: Some(Arc::new(initializer)),
            cache: (0..len).map(|_| OnceCell::new()).collect::<Vec<_>>().into(),
        }
    }

    /// Returns the number of children without initializing any slot.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns whether there are no children.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Returns a child, initializing and caching its slot on first access.
    pub fn get(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        let Some(cell) = self.cache.get(index) else {
            return Ok(None);
        };
        if let Some(child) = cell.get() {
            return Ok(Some(child.clone()));
        }

        let initializer = self
            .initializer
            .as_ref()
            .ok_or_else(|| vortex_err!("Plan child {index} was not initialized"))?;
        Ok(Some(cell.get_or_try_init(|| initializer(index))?.clone()))
    }

    /// Iterates over the children in logical order, initializing slots as they are visited.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = VortexResult<PlanRef>> + '_ {
        (0..self.len()).map(|index| {
            self.get(index)?
                .ok_or_else(|| vortex_err!("Plan child {index} is absent"))
        })
    }

    /// Materializes all children into an eager vector.
    pub fn to_vec(&self) -> VortexResult<Vec<PlanRef>> {
        self.iter().collect()
    }
}

impl From<Vec<PlanRef>> for PlanChildren {
    fn from(children: Vec<PlanRef>) -> Self {
        let cache = children
            .into_iter()
            .map(OnceCell::with_value)
            .collect::<Vec<_>>()
            .into();
        Self {
            initializer: None,
            cache,
        }
    }
}

impl<const N: usize> From<[PlanRef; N]> for PlanChildren {
    fn from(children: [PlanRef; N]) -> Self {
        Vec::from(children).into()
    }
}

impl Default for PlanChildren {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl fmt::Debug for PlanChildren {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlanChildren")
            .field("len", &self.len())
            .field(
                "initialized",
                &self
                    .cache
                    .iter()
                    .filter(|slot| slot.get().is_some())
                    .count(),
            )
            .finish()
    }
}
