// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`BindCache`]: a scan request bound once for all the sources of one schema.

use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::BoundExpression;
use vortex_expr::Expression;

use crate::prepare::SharedFilterPrep;

/// The most requests a cache holds; a scan has one, so more only come of a host reusing a cache.
const MAX_ENTRIES: usize = 8;

/// Scan requests bound to a dtype, shared by the scans of sources of that dtype, such as the files
/// of a multi-file scan, so that each request is bound, optimized and rewritten for pruning once
/// rather than once per source.
///
/// A host puts one in each [`ScanRequest`](crate::ScanRequest) of a scan over several sources. A
/// cache is cheap to clone: clones share their entries.
#[derive(Clone, Default)]
pub struct BindCache(Arc<Mutex<Vec<Arc<BoundRequest>>>>);

impl fmt::Debug for BindCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindCache")
            .field("entries", &self.0.lock().len())
            .finish()
    }
}

impl BindCache {
    /// The number of requests the cache holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.lock().len()
    }

    /// The request of `projection` and `filter` bound to `dtype`, binding it if no source of the
    /// cache has yet.
    pub(crate) fn bind(
        &self,
        dtype: &DType,
        projection: &Expression,
        filter: Option<&Expression>,
    ) -> VortexResult<Arc<BoundRequest>> {
        let found = self
            .0
            .lock()
            .iter()
            .find(|bound| bound.is_for(dtype, projection, filter))
            .cloned();
        if let Some(bound) = found {
            return Ok(bound);
        }
        let bound = Arc::new(BoundRequest::new(dtype, projection, filter)?);
        let mut entries = self.0.lock();
        if entries.len() >= MAX_ENTRIES {
            entries.remove(0);
        }
        entries.push(Arc::clone(&bound));
        Ok(bound)
    }
}

/// A scan request's projection and filter, bound to a dtype and optimized.
pub(crate) struct BoundRequest {
    dtype: DType,
    projection_expr: Expression,
    filter_expr: Option<Expression>,
    pub(crate) projection: BoundExpression,
    pub(crate) filter: Option<BoundExpression>,
    /// The filter's rewrites for its plans, built by the first scan that plans it.
    pub(crate) filter_prep: Arc<SharedFilterPrep>,
}

impl BoundRequest {
    /// Binds `projection` and `filter` to `dtype` and optimizes them.
    pub(crate) fn new(
        dtype: &DType,
        projection: &Expression,
        filter: Option<&Expression>,
    ) -> VortexResult<Self> {
        Ok(Self {
            dtype: dtype.clone(),
            projection_expr: projection.clone(),
            filter_expr: filter.cloned(),
            projection: projection.bind(dtype)?.optimize_recursive()?,
            filter: filter
                .map(|filter| filter.bind(dtype)?.optimize_recursive())
                .transpose()?,
            filter_prep: Arc::default(),
        })
    }

    fn is_for(&self, dtype: &DType, projection: &Expression, filter: Option<&Expression>) -> bool {
        &self.projection_expr == projection
            && self.filter_expr.as_ref() == filter
            && &self.dtype == dtype
    }
}
