// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod children;
pub mod layouts;

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::future::{BoxFuture, Shared};
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexResult};
use vortex_gpu::GpuVector;

pub type GpuLayoutReaderRef = Arc<dyn GpuLayoutReader>;

pub type GpuArrayFuture = BoxFuture<'static, VortexResult<Vec<GpuVector>>>;

pub type ShareGpuArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;

/// A [`crate::gpu::GpuLayoutReader`] is used to read a [`crate::Layout`] in a way that can cache state across multiple
/// evaluation operations.
pub trait GpuLayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the un-projected dtype of the layout reader.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the layout reader.
    /// An inexact count may be larger or smaller than the actual row count.
    fn row_count(&self) -> Precision<u64>;

    /// Register the splits of this layout reader.
    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
    ) -> VortexResult<GpuArrayFuture>;
}
