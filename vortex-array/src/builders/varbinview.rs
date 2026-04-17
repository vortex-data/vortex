// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::build_views::BinaryView;
use crate::arrays::varbinview::compact::BufferUtilization;
use crate::builders::ArrayBuilder;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
#[expect(deprecated)]
use crate::canonical::ToCanonical as _;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// The builder for building a [`VarBinViewArray`].
pub struct VarBinViewBuilder {
    dtype: DType,
    views_builder: BufferMut<BinaryView>,
    nulls: LazyBitBufferBuilder,
    completed: CompletedBuffers,
    in_progress: ByteBufferMut,
    growth_strategy: BufferGrowthStrategy,
    compaction_threshold: f64,
}

impl VarBinViewBuilder {
    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        Self::new(dtype, capacity, Default::default(), Default::default(), 0.0)
    }

    pub fn with_buffer_deduplication(dtype: DType, capacity: usize) -> Self {
        Self::new(
            dtype,
            capacity,
            CompletedBuffers::Deduplicated(Default::default()),
            Default::default(),
            0.0,
        )
    }

    pub fn with_compaction(dtype: DType, capacity: usize, compaction_threshold: f64) -> Self {
        Self::new(
            dtype,
            capacity,
            Default::default(),
            Default::default(),
            compaction_threshold,
        )
    }

    pub fn new(
        dtype: DType,
        capacity: usize,
        completed: CompletedBuffers,
        growth_strategy: BufferGrowthStrategy,
        compaction_threshold: f64,
    ) -> Self {
        assert!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "VarBinViewBuilder DType must be Utf8 or Binary."
        );
        Self {
            views_builder: BufferMut::<BinaryView>::with_capacity(capacity),
            nulls: LazyBitBufferBuilder::new(capacity),
            completed,
            in_progress: ByteBufferMut::empty(),
            dtype,
            growth_strategy,
            compaction_threshold,
        }
    }

    fn append_value_view(&mut self, value: &[u8]) {
        let length =
            u32::try_from(value.len()).vortex_expect("cannot have a single string >2^32 in length");
        if length <= 12 {
            self.views_builder.push(BinaryView::make_view(value, 0, 0));
            return;
        }

        let (buffer_idx, offset) = self.append_value_to_buffer(value);
        let view = BinaryView::make_view(value, buffer_idx, offset);
        self.views_builder.push(view);
    }

    /// Appends a value to the builder.
    pub fn append_value<S: AsRef<[u8]>>(&mut self, value: S) {
        self.append_value_view(value.as_ref());
        self.nulls.append_non_null();
    }

    /// Appends `n` copies of `value` as non-null entries.
    pub fn append_n_values<S: AsRef<[u8]>>(&mut self, value: S, n: usize) {
        if n == 0 {
            return;
        }
        let bytes = value.as_ref();
        let view = if bytes.len() <= BinaryView::MAX_INLINED_SIZE {
            BinaryView::make_view(bytes, 0, 0)
        } else {
            let (buffer_idx, offset) = self.append_value_to_buffer(bytes);
            BinaryView::make_view(bytes, buffer_idx, offset)
        };
        self.views_builder.push_n(view, n);
        self.nulls.append_n_non_nulls(n);
    }

    fn flush_in_progress(&mut self) {
        if self.in_progress.is_empty() {
            return;
        }
        let block = std::mem::take(&mut self.in_progress).freeze();

        assert!(block.len() < u32::MAX as usize, "Block too large");

        let initial_len = self.completed.len();
        self.completed.push(block);
        assert_eq!(
            self.completed.len(),
            initial_len + 1,
            "Invalid state, just completed block already exists"
        );
    }

    /// append a non inlined value to self.in_progress.
    fn append_value_to_buffer(&mut self, value: &[u8]) -> (u32, u32) {
        assert!(
            value.len() > BinaryView::MAX_INLINED_SIZE,
            "must inline small strings"
        );
        let required_cap = self.in_progress.len() + value.len();
        if self.in_progress.capacity() < required_cap {
            self.flush_in_progress();
            let next_buffer_size = self.growth_strategy.next_size() as usize;
            let to_reserve = next_buffer_size.max(value.len());
            self.in_progress.reserve(to_reserve);
        }

        let buffer_idx = self.completed.len();
        let offset = u32::try_from(self.in_progress.len()).vortex_expect("too many buffers");
        self.in_progress.extend_from_slice(value);

        (buffer_idx, offset)
    }

    pub fn completed_block_count(&self) -> u32 {
        self.completed.len()
    }

    /// Returns true if a non-empty in-progress buffer is staged (and would
    /// become a completed buffer on the next flush), false otherwise.
    pub fn in_progress(&self) -> bool {
        !self.in_progress.is_empty()
    }

    /// Pushes buffers and pre-adjusted views into the builder.
    ///
    /// The provided `buffers` contain sections of data from a `VarBinViewArray`, and the
    /// `views` are `BinaryView`s that have already been adjusted to reference the correct buffer
    /// indices and offsets for this builder. All views must point to valid sections within the
    /// provided buffers, and the validity length must match the view length.
    ///
    /// # Warning
    ///
    /// This method does not check utilization of the given buffers. Callers must provide
    /// buffers that are fully utilized by the given adjusted views.
    ///
    /// # Panics
    ///
    /// Panics if this builder deduplicates buffers and any of the given buffers already
    /// exist in this builder.
    pub fn push_buffer_and_adjusted_views(
        &mut self,
        buffers: &[ByteBuffer],
        views: &Buffer<BinaryView>,
        validity_mask: Mask,
    ) {
        self.flush_in_progress();

        let expected_completed_len = self.completed.len() as usize + buffers.len();
        self.completed.extend_from_slice_unchecked(buffers);
        assert_eq!(
            self.completed.len() as usize,
            expected_completed_len,
            "Some buffers already exist",
        );
        self.views_builder.extend_trusted(views.iter().copied());
        self.push_only_validity_mask(validity_mask);

        debug_assert_eq!(self.nulls.len(), self.views_builder.len())
    }

    /// Finishes the builder directly into a [`VarBinViewArray`].
    pub fn finish_into_varbinview(&mut self) -> VarBinViewArray {
        self.flush_in_progress();
        let buffers = std::mem::take(&mut self.completed);

        assert_eq!(
            self.views_builder.len(),
            self.nulls.len(),
            "View and validity length must match"
        );

        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        // SAFETY: the builder methods check safety at each step.
        unsafe {
            VarBinViewArray::new_unchecked(
                std::mem::take(&mut self.views_builder).freeze(),
                buffers.finish(),
                self.dtype.clone(),
                validity,
            )
        }
    }

    // Pushes a validity mask into the builder not affecting the views or buffers
    fn push_only_validity_mask(&mut self, validity_mask: Mask) {
        self.nulls.append_validity_mask(validity_mask);
    }
}

impl ArrayBuilder for VarBinViewBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.nulls.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "VarBinViewBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        match self.dtype() {
            DType::Utf8(_) => match scalar.as_utf8().value() {
                Some(value) => self.append_value(value),
                None => self.append_null(),
            },
            DType::Binary(_) => match scalar.as_binary().value() {
                Some(value) => self.append_value(value),
                None => self.append_null(),
            },
            _ => vortex_bail!(
                "VarBinViewBuilder can only handle Utf8 or Binary scalars, got {:?}",
                scalar.dtype()
            ),
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        #[expect(deprecated)]
        let array = array.to_varbinview();
        self.flush_in_progress();

        self.push_only_validity_mask(
            array
                .as_ref()
                .validity()
                .vortex_expect("validity_mask")
                .to_mask(
                    array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .vortex_expect("Failed to compute validity mask"),
        );

        let view_adjustment =
            self.completed
                .extend_from_compaction(BuffersWithOffsets::from_array(
                    &array,
                    self.compaction_threshold,
                ));

        match view_adjustment {
            ViewAdjustment::Precomputed(adjustment) => self.views_builder.extend_trusted(
                array
                    .views()
                    .iter()
                    .map(|view| adjustment.adjust_view(view)),
            ),
            ViewAdjustment::Rewriting(adjustment) => {
                match array
                    .as_ref()
                    .validity()
                    .vortex_expect("validity_mask")
                    .to_mask(
                        array.as_ref().len(),
                        &mut LEGACY_SESSION.create_execution_ctx(),
                    )
                    .vortex_expect("Failed to compute validity mask")
                {
                    Mask::AllTrue(_) => {
                        for (idx, &view) in array.views().iter().enumerate() {
                            let new_view = self.push_view(view, &adjustment, &array, idx);
                            self.views_builder.push(new_view);
                        }
                    }
                    Mask::AllFalse(_) => {
                        self.views_builder
                            .push_n(BinaryView::empty_view(), array.len());
                    }
                    Mask::Values(v) => {
                        for (idx, (&view, is_valid)) in
                            array.views().iter().zip(v.bit_buffer().iter()).enumerate()
                        {
                            let new_view = if !is_valid {
                                BinaryView::empty_view()
                            } else {
                                self.push_view(view, &adjustment, &array, idx)
                            };
                            self.views_builder.push(new_view);
                        }
                    }
                }
            }
        }
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.views_builder.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_varbinview().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::VarBinView(self.finish_into_varbinview())
    }
}

impl VarBinViewBuilder {
    fn push_view(
        &mut self,
        view: BinaryView,
        adjustment: &RewritingViewAdjustment,
        array: &VarBinViewArray,
        idx: usize,
    ) -> BinaryView {
        if view.is_inlined() {
            view
        } else if let Some(adjusted) = adjustment.adjust_view(&view) {
            adjusted
        } else {
            let bytes = array.bytes_at(idx);
            let (new_buf_idx, new_offset) = self.append_value_to_buffer(&bytes);
            BinaryView::make_view(bytes.as_slice(), new_buf_idx, new_offset)
        }
    }
}

pub enum CompletedBuffers {
    Default(Vec<ByteBuffer>),
    Deduplicated(DeduplicatedBuffers),
}

impl Default for CompletedBuffers {
    fn default() -> Self {
        Self::Default(Vec::new())
    }
}

// Self::push enforces len < u32::max
#[expect(clippy::cast_possible_truncation)]
impl CompletedBuffers {
    fn len(&self) -> u32 {
        match self {
            Self::Default(buffers) => buffers.len() as u32,
            Self::Deduplicated(buffers) => buffers.len(),
        }
    }

    fn push(&mut self, block: ByteBuffer) -> u32 {
        match self {
            Self::Default(buffers) => {
                assert!(buffers.len() < u32::MAX as usize, "Too many blocks");
                buffers.push(block);
                self.len()
            }
            Self::Deduplicated(buffers) => buffers.push(block),
        }
    }

    /// Does not compact buffers, bypasses utilization checks.
    fn extend_from_slice_unchecked(&mut self, buffers: &[ByteBuffer]) {
        for buffer in buffers {
            self.push(buffer.clone());
        }
    }

    fn extend_from_compaction(&mut self, buffers: BuffersWithOffsets) -> ViewAdjustment {
        match (self, buffers) {
            (
                Self::Default(completed_buffers),
                BuffersWithOffsets::AllKept { buffers, offsets },
            ) => {
                let buffer_offset = completed_buffers.len() as u32;
                completed_buffers.extend_from_slice(&buffers);
                ViewAdjustment::shift(buffer_offset, offsets)
            }
            (
                Self::Default(completed_buffers),
                BuffersWithOffsets::SomeCompacted { buffers, offsets },
            ) => {
                let lookup = buffers
                    .iter()
                    .map(|maybe_buffer| {
                        maybe_buffer.as_ref().map(|buffer| {
                            completed_buffers.push(buffer.clone());
                            completed_buffers.len() as u32 - 1
                        })
                    })
                    .collect();
                ViewAdjustment::rewriting(lookup, offsets)
            }

            (
                Self::Deduplicated(completed_buffers),
                BuffersWithOffsets::AllKept { buffers, offsets },
            ) => {
                let buffer_lookup = completed_buffers.extend_from_iter(buffers.iter().cloned());
                ViewAdjustment::lookup(buffer_lookup, offsets)
            }
            (
                Self::Deduplicated(completed_buffers),
                BuffersWithOffsets::SomeCompacted { buffers, offsets },
            ) => {
                let buffer_lookup = completed_buffers.extend_from_option_slice(&buffers);
                ViewAdjustment::rewriting(buffer_lookup, offsets)
            }
        }
    }

    fn finish(self) -> Arc<[ByteBuffer]> {
        match self {
            Self::Default(buffers) => Arc::from(buffers),
            Self::Deduplicated(buffers) => buffers.finish(),
        }
    }
}

#[derive(Default)]
pub struct DeduplicatedBuffers {
    buffers: Vec<ByteBuffer>,
    buffer_to_idx: HashMap<BufferId, u32>,
}

impl DeduplicatedBuffers {
    // Self::push enforces len < u32::max
    #[expect(clippy::cast_possible_truncation)]
    fn len(&self) -> u32 {
        self.buffers.len() as u32
    }

    /// Push a new block if not seen before. Returns the idx of the block.
    pub(crate) fn push(&mut self, block: ByteBuffer) -> u32 {
        assert!(self.buffers.len() < u32::MAX as usize, "Too many blocks");

        let initial_len = self.len();
        let id = BufferId::from(&block);
        match self.buffer_to_idx.entry(id) {
            Entry::Occupied(idx) => *idx.get(),
            Entry::Vacant(entry) => {
                let idx = initial_len;
                entry.insert(idx);
                self.buffers.push(block);
                idx
            }
        }
    }

    pub(crate) fn extend_from_option_slice(
        &mut self,
        buffers: &[Option<ByteBuffer>],
    ) -> Vec<Option<u32>> {
        buffers
            .iter()
            .map(|buffer| buffer.as_ref().map(|buf| self.push(buf.clone())))
            .collect()
    }

    pub(crate) fn extend_from_iter(
        &mut self,
        buffers: impl Iterator<Item = ByteBuffer>,
    ) -> Vec<u32> {
        buffers.map(|buffer| self.push(buffer)).collect()
    }

    pub(crate) fn finish(self) -> Arc<[ByteBuffer]> {
        Arc::from(self.buffers)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct BufferId {
    // *const u8 stored as usize for `Send`
    ptr: usize,
    len: usize,
}

impl BufferId {
    fn from(buffer: &ByteBuffer) -> Self {
        let slice = buffer.as_slice();
        Self {
            ptr: slice.as_ptr() as usize,
            len: slice.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BufferGrowthStrategy {
    /// Use a fixed buffer size for all allocations.
    Fixed { size: u32 },
    /// Use exponential growth starting from initial_size, doubling until max_size.
    Exponential { current_size: u32, max_size: u32 },
}

impl Default for BufferGrowthStrategy {
    fn default() -> Self {
        Self::Exponential {
            current_size: 4 * 1024,    // 4KB starting size
            max_size: 2 * 1024 * 1024, // 2MB max size
        }
    }
}

impl BufferGrowthStrategy {
    pub fn fixed(size: u32) -> Self {
        Self::Fixed { size }
    }

    pub fn exponential(initial_size: u32, max_size: u32) -> Self {
        Self::Exponential {
            current_size: initial_size,
            max_size,
        }
    }

    /// Returns the next buffer size to allocate and updates internal state.
    pub fn next_size(&mut self) -> u32 {
        match self {
            Self::Fixed { size } => *size,
            Self::Exponential {
                current_size,
                max_size,
            } => {
                let result = *current_size;
                if *current_size < *max_size {
                    *current_size = current_size.saturating_mul(2).min(*max_size);
                }
                result
            }
        }
    }
}

enum BuffersWithOffsets {
    AllKept {
        buffers: Arc<[ByteBuffer]>,
        offsets: Option<Vec<u32>>,
    },
    SomeCompacted {
        buffers: Vec<Option<ByteBuffer>>,
        offsets: Option<Vec<u32>>,
    },
}

impl BuffersWithOffsets {
    pub fn from_array(array: &VarBinViewArray, compaction_threshold: f64) -> Self {
        if compaction_threshold == 0.0 {
            return Self::AllKept {
                buffers: Arc::from(
                    array
                        .data_buffers()
                        .to_vec()
                        .into_iter()
                        .map(|b| b.unwrap_host())
                        .collect_vec(),
                ),
                offsets: None,
            };
        }

        let buffer_utilizations = array
            .buffer_utilizations()
            .vortex_expect("buffer_utilizations in BuffersWithOffsets::from_array");
        let mut has_rewrite = false;
        let mut has_nonzero_offset = false;
        for utilization in buffer_utilizations.iter() {
            match compaction_strategy(utilization, compaction_threshold) {
                CompactionStrategy::KeepFull => continue,
                CompactionStrategy::Slice { .. } => has_nonzero_offset = true,
                CompactionStrategy::Rewrite => has_rewrite = true,
            }
        }

        let buffers_with_offsets_iter = buffer_utilizations
            .iter()
            .zip(array.data_buffers().iter())
            .map(|(utilization, buffer)| {
                match compaction_strategy(utilization, compaction_threshold) {
                    CompactionStrategy::KeepFull => (Some(buffer.as_host().clone()), 0),
                    CompactionStrategy::Slice { start, end } => (
                        Some(buffer.as_host().slice(start as usize..end as usize)),
                        start,
                    ),
                    CompactionStrategy::Rewrite => (None, 0),
                }
            });

        match (has_rewrite, has_nonzero_offset) {
            // keep all buffers
            (false, false) => {
                let buffers: Vec<_> = buffers_with_offsets_iter
                    .map(|(b, _)| b.vortex_expect("already checked for rewrite"))
                    .collect();
                Self::AllKept {
                    buffers: Arc::from(buffers),
                    offsets: None,
                }
            }
            // rewrite, all zero offsets
            (true, false) => {
                let buffers: Vec<_> = buffers_with_offsets_iter.map(|(b, _)| b).collect();
                Self::SomeCompacted {
                    buffers,
                    offsets: None,
                }
            }
            // keep all buffers, but some have offsets
            (false, true) => {
                let (buffers, offsets): (Vec<_>, _) = buffers_with_offsets_iter
                    .map(|(buffer, offset)| {
                        (buffer.vortex_expect("already checked for rewrite"), offset)
                    })
                    .collect();
                Self::AllKept {
                    buffers: Arc::from(buffers),
                    offsets: Some(offsets),
                }
            }
            // rewrite and some have offsets
            (true, true) => {
                let (buffers, offsets) = buffers_with_offsets_iter.collect();
                Self::SomeCompacted {
                    buffers,
                    offsets: Some(offsets),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompactionStrategy {
    KeepFull,
    /// Slice the buffer to [start, end) range
    Slice {
        start: u32,
        end: u32,
    },
    /// Rewrite data into new compacted buffer
    Rewrite,
}

fn compaction_strategy(
    buffer_utilization: &BufferUtilization,
    threshold: f64,
) -> CompactionStrategy {
    match buffer_utilization.overall_utilization() {
        // rewrite empty or not used buffers TODO(os): maybe keep them
        0.0 => CompactionStrategy::Rewrite,
        utilised if utilised >= threshold => CompactionStrategy::KeepFull,
        _ if buffer_utilization.range_utilization() >= threshold => {
            let Range { start, end } = buffer_utilization.range();
            CompactionStrategy::Slice { start, end }
        }
        _ => CompactionStrategy::Rewrite,
    }
}

enum ViewAdjustment {
    Precomputed(PrecomputedViewAdjustment),
    Rewriting(RewritingViewAdjustment),
}

impl ViewAdjustment {
    fn shift(buffer_offset: u32, offsets: Option<Vec<u32>>) -> Self {
        Self::Precomputed(PrecomputedViewAdjustment::Shift {
            buffer_offset,
            offsets,
        })
    }

    fn lookup(buffer_lookup: Vec<u32>, offsets: Option<Vec<u32>>) -> Self {
        Self::Precomputed(PrecomputedViewAdjustment::Lookup {
            buffer_lookup,
            offsets,
        })
    }

    fn rewriting(buffer_lookup: Vec<Option<u32>>, offsets: Option<Vec<u32>>) -> Self {
        Self::Rewriting(RewritingViewAdjustment {
            buffer_lookup,
            offsets,
        })
    }
}

// Care when adding new variants or fields in this enum, it will mess with inlining if it gets too big
enum PrecomputedViewAdjustment {
    Shift {
        buffer_offset: u32,
        offsets: Option<Vec<u32>>,
    },
    Lookup {
        buffer_lookup: Vec<u32>,
        offsets: Option<Vec<u32>>,
    },
}

impl PrecomputedViewAdjustment {
    fn adjust_view(&self, view: &BinaryView) -> BinaryView {
        if view.is_inlined() {
            return *view;
        }
        let view_ref = view.as_view();
        match self {
            Self::Shift {
                buffer_offset,
                offsets,
            } => {
                let b_idx = view_ref.buffer_index;
                let offset_shift = offsets
                    .as_ref()
                    .map(|o| o[b_idx as usize])
                    .unwrap_or_default();

                // If offset < offset_shift, this view was invalid and wasn't counted in buffer_utilizations.
                // Return an empty view to match how invalid views are handled in the Rewriting path.
                if view_ref.offset < offset_shift {
                    return BinaryView::empty_view();
                }

                view_ref
                    .with_buffer_and_offset(b_idx + buffer_offset, view_ref.offset - offset_shift)
            }
            Self::Lookup {
                buffer_lookup,
                offsets,
            } => {
                let b_idx = view_ref.buffer_index;
                let buffer = buffer_lookup[b_idx as usize];
                let offset_shift = offsets
                    .as_ref()
                    .map(|o| o[b_idx as usize])
                    .unwrap_or_default();

                // If offset < offset_shift, this view was invalid and wasn't counted in buffer_utilizations.
                // Return an empty view to match how invalid views are handled in the Rewriting path.
                if view_ref.offset < offset_shift {
                    return BinaryView::empty_view();
                }

                view_ref.with_buffer_and_offset(buffer, view_ref.offset - offset_shift)
            }
        }
        .into()
    }
}

struct RewritingViewAdjustment {
    buffer_lookup: Vec<Option<u32>>,
    offsets: Option<Vec<u32>>,
}

impl RewritingViewAdjustment {
    /// Can return None if this view can't be adjusted, because there is no precomputed lookup
    /// for the current buffer.
    fn adjust_view(&self, view: &BinaryView) -> Option<BinaryView> {
        if view.is_inlined() {
            return Some(*view);
        }

        let view_ref = view.as_view();
        self.buffer_lookup[view_ref.buffer_index as usize].map(|buffer| {
            let offset_shift = self
                .offsets
                .as_ref()
                .map(|o| o[view_ref.buffer_index as usize])
                .unwrap_or_default();
            view_ref
                .with_buffer_and_offset(buffer, view_ref.offset - offset_shift)
                .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::VarBinViewBuilder;
    use crate::builders::varbinview::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn test_utf8_builder() {
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        builder.append_value("Hello");
        builder.append_null();
        builder.append_value("World");

        builder.append_nulls(2);

        builder.append_zeros(2);
        builder.append_value("test");

        let actual = builder.finish();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("Hello"),
            None,
            Some("World"),
            None,
            None,
            Some(""),
            Some(""),
            Some("test"),
        ]);
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_utf8_builder_with_extend() {
        let array = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_null();
            builder.append_value("Hello2");
            builder.finish()
        };
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        builder.append_value("Hello1");
        builder.extend_from_array(&array);
        builder.append_nulls(2);
        builder.append_value("Hello3");

        let actual = builder.finish_into_canonical();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("Hello1"),
            None,
            Some("Hello2"),
            None,
            None,
            Some("Hello3"),
        ]);
        assert_arrays_eq!(actual.into_array(), expected.into_array());
    }

    #[test]
    fn test_buffer_deduplication() -> VortexResult<()> {
        let array = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_value("This is a long string that should not be inlined");
            builder.append_value("short string");
            builder.finish_into_varbinview()
        };

        assert_eq!(array.data_buffers().len(), 1);
        let mut builder =
            VarBinViewBuilder::with_buffer_deduplication(DType::Utf8(Nullability::Nullable), 10);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        array.append_to_builder(&mut builder, &mut ctx)?;
        assert_eq!(builder.completed_block_count(), 1);

        array
            .slice(1..2)?
            .append_to_builder(&mut builder, &mut ctx)?;
        array
            .slice(0..1)?
            .append_to_builder(&mut builder, &mut ctx)?;
        assert_eq!(builder.completed_block_count(), 1);

        let array2 = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_value("This is a long string that should not be inlined");
            builder.finish_into_varbinview()
        };

        array2.append_to_builder(&mut builder, &mut ctx)?;
        assert_eq!(builder.completed_block_count(), 2);

        array
            .slice(0..1)?
            .append_to_builder(&mut builder, &mut ctx)?;
        array2
            .slice(0..1)?
            .append_to_builder(&mut builder, &mut ctx)?;
        assert_eq!(builder.completed_block_count(), 2);
        Ok(())
    }

    #[test]
    fn test_append_scalar() {
        use crate::scalar::Scalar;

        // Test with Utf8 builder.
        let mut utf8_builder =
            VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        // Test appending a valid utf8 value.
        let utf8_scalar1 = Scalar::utf8("hello", Nullability::Nullable);
        utf8_builder.append_scalar(&utf8_scalar1).unwrap();

        // Test appending another value.
        let utf8_scalar2 = Scalar::utf8("world", Nullability::Nullable);
        utf8_builder.append_scalar(&utf8_scalar2).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(DType::Utf8(Nullability::Nullable));
        utf8_builder.append_scalar(&null_scalar).unwrap();

        let array = utf8_builder.finish();
        let expected =
            <VarBinViewArray as FromIterator<_>>::from_iter([Some("hello"), Some("world"), None]);
        assert_arrays_eq!(&array, &expected);

        // Test with Binary builder.
        let mut binary_builder =
            VarBinViewBuilder::with_capacity(DType::Binary(Nullability::Nullable), 10);

        let binary_scalar = Scalar::binary(vec![1u8, 2, 3], Nullability::Nullable);
        binary_builder.append_scalar(&binary_scalar).unwrap();

        let binary_null = Scalar::null(DType::Binary(Nullability::Nullable));
        binary_builder.append_scalar(&binary_null).unwrap();

        let binary_array = binary_builder.finish();
        let expected =
            <VarBinViewArray as FromIterator<_>>::from_iter([Some(vec![1u8, 2, 3]), None]);
        assert_arrays_eq!(&binary_array, &expected);

        // Test wrong dtype error.
        let mut builder =
            VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }

    #[test]
    fn test_buffer_growth_strategies() {
        use super::BufferGrowthStrategy;

        // Test Fixed strategy
        let mut strategy = BufferGrowthStrategy::fixed(1024);

        // Should always return the fixed size
        assert_eq!(strategy.next_size(), 1024);
        assert_eq!(strategy.next_size(), 1024);
        assert_eq!(strategy.next_size(), 1024);

        // Test Exponential strategy
        let mut strategy = BufferGrowthStrategy::exponential(1024, 8192);

        // Should double each time until hitting max_size
        assert_eq!(strategy.next_size(), 1024); // First: 1024
        assert_eq!(strategy.next_size(), 2048); // Second: 2048
        assert_eq!(strategy.next_size(), 4096); // Third: 4096
        assert_eq!(strategy.next_size(), 8192); // Fourth: 8192 (max)
        assert_eq!(strategy.next_size(), 8192); // Fifth: 8192 (capped)
    }

    #[test]
    fn test_large_value_allocation() {
        use super::BufferGrowthStrategy;
        use super::VarBinViewBuilder;

        let mut builder = VarBinViewBuilder::new(
            DType::Binary(Nullability::Nullable),
            10,
            Default::default(),
            BufferGrowthStrategy::exponential(1024, 4096),
            0.0,
        );

        // Create a value larger than max_size
        let large_value = vec![0u8; 8192];

        // Should successfully append the large value
        builder.append_value(&large_value);

        let array = builder.finish_into_varbinview();
        assert_eq!(array.len(), 1);

        // Verify the value was stored correctly
        let retrieved = array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .as_binary()
            .value()
            .cloned()
            .unwrap();
        assert_eq!(retrieved.len(), 8192);
        assert_eq!(retrieved.as_slice(), &large_value);
    }
}
