// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::task::Poll;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;
use crate::session::LayoutSession;

pub fn new_session() -> VortexSession {
    array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

pub fn session_with_handle(handle: Handle) -> VortexSession {
    new_session().with_handle(handle)
}

pub static SCAN_SESSION: LazyLock<VortexSession> = LazyLock::new(new_session);

/// A configurable [`LayoutReader`] test double producing `base + row` for every selected row.
///
/// `split_size` controls the split layout (`None` is a single split), `keep_row` filters rows, and
/// the `fail_*` flags inject filter/projection errors. Every projection records its mask's
/// true-count into `projection_masks`, letting tests assert that a limit is applied before
/// projection.
#[derive(Debug)]
pub struct TestLayoutReader {
    name: Arc<str>,
    dtype: DType,
    row_count: u64,
    base: i32,
    split_size: Option<u64>,
    keep_row: fn(u64) -> bool,
    fail_first_filter: bool,
    fail_first_projection: bool,
    fail_projection: bool,
    projection_masks: Option<Arc<Mutex<Vec<usize>>>>,
}

impl TestLayoutReader {
    pub fn new(row_count: u64) -> Self {
        Self {
            name: Arc::from("test"),
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            row_count,
            base: 0,
            split_size: None,
            keep_row: keep_all,
            fail_first_filter: false,
            fail_first_projection: false,
            fail_projection: false,
            projection_masks: None,
        }
    }

    /// Offset every produced value, so tests can tell which reader a row came from.
    pub fn with_base(mut self, base: i32) -> Self {
        self.base = base;
        self
    }

    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.split_size = Some(split_size);
        self
    }

    pub fn with_keep_row(mut self, keep_row: fn(u64) -> bool) -> Self {
        self.keep_row = keep_row;
        self
    }

    pub fn with_projection_masks(mut self, projection_masks: Arc<Mutex<Vec<usize>>>) -> Self {
        self.projection_masks = Some(projection_masks);
        self
    }

    pub fn with_fail_first_filter(mut self) -> Self {
        self.fail_first_filter = true;
        self
    }

    pub fn with_fail_first_projection(mut self) -> Self {
        self.fail_first_projection = true;
        self
    }

    pub fn with_projection_error(mut self) -> Self {
        self.fail_projection = true;
        self
    }
}

impl LayoutReader for TestLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        let row_range = split_range.row_range();
        if let Some(size) = self.split_size {
            let mut boundary = row_range.start + size;
            while boundary < row_range.end {
                splits.push(split_range.row_offset() + boundary);
                boundary += size;
            }
        }
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        _expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if self.fail_first_filter && row_range.start == 0 {
            let len = mask.len();
            return Ok(MaskFuture::new(len, async move {
                Err(vortex_err!("first split filter failed"))
            }));
        }

        let row_range = row_range.clone();
        let keep_row = self.keep_row;
        let row_count = usize::try_from(row_range.end - row_range.start)
            .map_err(|_| vortex_err!("row range must fit in usize"))?;

        Ok(MaskFuture::new(row_count, async move {
            let input_mask = mask.await?;
            Ok(Mask::from_iter(
                (row_range.start..row_range.end)
                    .enumerate()
                    .map(|(idx, row)| input_mask.value(idx) && keep_row(row)),
            ))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        _expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let row_range = row_range.clone();
        let base = self.base;
        let projection_masks = self.projection_masks.clone();
        let fail = self.fail_projection || (self.fail_first_projection && row_range.start == 0);

        Ok(Box::pin(async move {
            let mask = mask.await?;
            if let Some(projection_masks) = projection_masks {
                projection_masks.lock().push(mask.true_count());
            }
            if fail {
                return Err(vortex_err!("projection failed"));
            }
            let start = i32::try_from(row_range.start)
                .map_err(|_| vortex_err!("row_range.start must fit in i32"))?;
            let end = i32::try_from(row_range.end)
                .map_err(|_| vortex_err!("row_range.end must fit in i32"))?;
            PrimitiveArray::from_iter((start..end).map(|value| base + value))
                .into_array()
                .filter(mask)
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub fn keep_all(_: u64) -> bool {
    true
}

pub fn keep_odd(row: u64) -> bool {
    row % 2 == 1
}

/// Canonicalize every chunk of a scan into a flat list of `i32` values.
pub fn collect_scan_values<I>(iter: I) -> VortexResult<Vec<i32>>
where
    I: IntoIterator<Item = VortexResult<ArrayRef>>,
{
    let mut ctx = array_session().create_execution_ctx();
    let mut values = Vec::new();
    for chunk in iter {
        let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
        values.extend(primitive.into_buffer::<i32>());
    }
    Ok(values)
}

/// Let already-spawned scan tasks run to completion after the stream has been dropped.
pub fn drain_runtime(runtime: &SingleThreadRuntime) {
    for _ in 0..4 {
        let mut yielded = false;
        runtime.block_on(futures::future::poll_fn(move |cx| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }));
    }
}
