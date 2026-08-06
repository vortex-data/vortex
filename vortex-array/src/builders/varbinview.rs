// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Alignment;
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
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::arrays::varbinview::build_views::BinaryView;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::arrays::varbinview::build_views::extend_views;
use crate::arrays::varbinview::compact::BufferUtilization;
use crate::builders::ArrayBuilder;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::scalar::Scalar;

/// The builder for building a [`VarBinViewArray`].
pub struct VarBinViewBuilder {
    dtype: DType,
    views_builder: BufferMut<BinaryView>,
    nulls: LazyBitBufferBuilder,
    completed: CompletedBuffers,
    in_progress: Option<ByteBufferMut>,
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
            views_builder: BufferMut::with_capacity_preferred_aligned(
                capacity,
                Alignment::of::<BinaryView>(),
                None,
            ),
            nulls: LazyBitBufferBuilder::new(capacity),
            completed,
            in_progress: None,
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
        let Some(block) = self.in_progress.take() else {
            return;
        };

        assert!(block.len() < u32::MAX as usize, "Block too large");

        let initial_len = self.completed.len();
        self.completed.push(block.freeze());
        assert_eq!(
            self.completed.len(),
            initial_len + 1,
            "Invalid state, just completed block already exists"
        );
    }

    fn init_in_progress(&mut self, min_len: usize) {
        let next_buffer_size = self.growth_strategy.next_size() as usize;
        let to_reserve = next_buffer_size.max(min_len);
        self.in_progress = Some(ByteBufferMut::with_capacity_preferred_aligned(
            to_reserve,
            Alignment::of::<u8>(),
            None,
        ));
    }

    /// append a non inlined value to self.in_progress.
    fn append_value_to_buffer(&mut self, value: &[u8]) -> (u32, u32) {
        assert!(
            value.len() > BinaryView::MAX_INLINED_SIZE,
            "must inline small strings"
        );

        if let Some(in_progress) = &mut self.in_progress {
            let required_cap = in_progress.len() + value.len();
            if in_progress.capacity() < required_cap {
                self.flush_in_progress();
                self.init_in_progress(value.len());
            }
        } else {
            self.init_in_progress(value.len())
        };

        let in_progress = self
            .in_progress
            .as_mut()
            .vortex_expect("in_progress just set");

        let buffer_idx = self.completed.len();
        let offset = u32::try_from(in_progress.len()).vortex_expect("too many buffers");
        in_progress.extend_from_slice(value);

        (buffer_idx, offset)
    }

    pub fn completed_block_count(&self) -> u32 {
        self.completed.len()
    }

    /// Whether this builder compacts the data buffers it is handed. The lengths-driven appends
    /// use this to gate their utilization measurement; the buffer-adopting escape hatch
    /// ([`append_views_built_at`](Self::append_views_built_at)) always bypasses it.
    fn compacts_buffers(&self) -> bool {
        self.compaction_threshold > 0.0
    }

    /// Adopts the buffers and views that `build` produces against the index its first buffer
    /// will land at.
    ///
    /// The builder flushes its staged bytes, then hands `build` the index the next data buffer
    /// will occupy; `build` returns data buffers — which land contiguously from that index — and
    /// one view per entry of `validity`, already referencing them. This is the escape hatch for
    /// an encoding that only discovers its views while walking its own byte format (e.g.
    /// length-prefixed frames), where the lengths-driven appends cannot apply; keeping the
    /// numbering inside this call is what makes the views come out right without a rebase pass.
    ///
    /// # Warning
    ///
    /// This method does not check utilization of the returned buffers. `build` must return
    /// buffers that are fully utilized by its views.
    ///
    /// # Panics
    ///
    /// Panics if `build` returns a different view count than `validity.len()`, or if this
    /// builder deduplicates buffers and already holds one of the returned buffers.
    pub fn append_views_built_at(
        &mut self,
        validity: &Mask,
        build: impl FnOnce(u32) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)>,
    ) -> VortexResult<()> {
        self.flush_in_progress();

        let start_index = self.completed.len();
        let (buffers, views) = build(start_index)?;
        assert_eq!(
            views.len(),
            validity.len(),
            "Must build one view per validity entry"
        );

        let expected_completed_len = start_index as usize + buffers.len();
        self.completed.extend_from_slice_unchecked(&buffers);
        assert_eq!(
            self.completed.len() as usize,
            expected_completed_len,
            "Some buffers already exist",
        );
        self.views_builder.extend_trusted(views.iter().copied());
        self.push_only_validity_mask(validity);

        debug_assert_eq!(self.nulls.len(), self.views_builder.len());
        Ok(())
    }

    /// Appends values laid end-to-end in `bytes`, one per entry of `lengths`.
    ///
    /// The builder adopts `bytes` as a data buffer without copying it (splitting it only past the
    /// `u32` view-offset limit) and builds the views directly into its own storage, so the whole
    /// append costs one view per row. Null rows must be zero-length entries of `lengths` — every
    /// length is consumed either way, so the lengths must describe `bytes` exactly.
    ///
    /// When the builder is configured to compact buffers, the utilization is measured from the
    /// lengths alone — only values too long to inline reference the buffer — and a heap below
    /// the threshold is rewritten to just those values instead of adopted, so callers never need
    /// a canonicalize-and-compact fallback.
    ///
    /// # Panics
    ///
    /// Panics if `lengths` and `validity` disagree in length, if the lengths do not describe
    /// `bytes` exactly, or if this builder deduplicates buffers and already holds `bytes`.
    pub fn append_buffer_with_lengths<P: NativePType + AsPrimitive<usize>>(
        &mut self,
        bytes: ByteBuffer,
        lengths: &[P],
        validity: &Mask,
    ) {
        assert_eq!(
            lengths.len(),
            validity.len(),
            "Must have one length per validity entry"
        );
        self.append_buffer_views(&bytes, lengths.len(), validity, |i| lengths[i].as_());
    }

    /// [`append_buffer_with_lengths`](Self::append_buffer_with_lengths) for values described by
    /// an offsets buffer instead of lengths.
    ///
    /// `offsets` are absolute positions into `bytes` — the layout a
    /// [`VarBinArray`](crate::arrays::VarBinArray) stores — so there is one more offset than there
    /// are values, and only the `offsets[0]..offsets[last]` range of `bytes` is adopted, again
    /// without copying.
    ///
    /// # Panics
    ///
    /// Panics if `offsets` does not hold exactly one more entry than `validity`, or if the offsets
    /// are not monotonically non-decreasing positions within `bytes`.
    pub fn append_buffer_with_offsets<P: NativePType + AsPrimitive<usize>>(
        &mut self,
        bytes: ByteBuffer,
        offsets: &[P],
        validity: &Mask,
    ) {
        assert_eq!(
            offsets.len(),
            validity.len() + 1,
            "Must have one more offset than validity entries"
        );
        let first: usize = offsets[0].as_();
        let last: usize = offsets[offsets.len() - 1].as_();
        let bytes = bytes.slice(first..last);
        // Wrapping keeps corrupt non-monotonic offsets from panicking on the subtraction itself;
        // the wrapped length then fails the in-bounds checks of the view-building loop.
        self.append_buffer_views(&bytes, validity.len(), validity, |i| {
            AsPrimitive::<usize>::as_(offsets[i + 1])
                .wrapping_sub(AsPrimitive::<usize>::as_(offsets[i]))
        });
    }

    /// Shared tail of the bulk buffer appends: builds the views straight into the builder's views
    /// storage, then adopts the buffer segments and the validity.
    fn append_buffer_views(
        &mut self,
        bytes: &ByteBuffer,
        count: usize,
        validity: &Mask,
        len_at: impl Fn(usize) -> usize,
    ) {
        self.flush_in_progress();

        // A compacting builder measures utilization before adopting the buffer. Only values too
        // long to inline reference the heap, so the measurement is one pass over the lengths and
        // never touches the bytes. Heaps past the single-buffer limit are adopted as they are:
        // they roll over into multiple buffers, and per-segment accounting is not worth the rare
        // >2GiB case.
        if self.compacts_buffers() && bytes.len() <= MAX_BUFFER_LEN {
            let referenced: usize = (0..count)
                .map(&len_at)
                .filter(|len| *len > BinaryView::MAX_INLINED_SIZE)
                .sum();
            #[expect(clippy::cast_precision_loss)]
            if (referenced as f64) < self.compaction_threshold * (bytes.len() as f64) {
                return self
                    .append_buffer_views_rewritten(bytes, count, validity, len_at, referenced);
            }
        }

        let start_index = self.completed.len();
        let segments = extend_views(
            &mut self.views_builder,
            start_index,
            MAX_BUFFER_LEN,
            bytes,
            count,
            len_at,
        );

        let expected_completed_len = start_index as usize + segments.len();
        self.completed.extend_from_slice_unchecked(&segments);
        assert_eq!(
            self.completed.len() as usize,
            expected_completed_len,
            "Some buffers already exist",
        );

        self.push_only_validity_mask(validity);
        debug_assert_eq!(self.nulls.len(), self.views_builder.len());
    }

    /// The under-utilized arm of [`append_buffer_views`](Self::append_buffer_views): copies only
    /// the values that actually reference the heap into a compact buffer, sized `referenced`,
    /// instead of adopting the whole heap. A fully-inlined append pushes no buffer at all.
    fn append_buffer_views_rewritten(
        &mut self,
        bytes: &ByteBuffer,
        count: usize,
        validity: &Mask,
        len_at: impl Fn(usize) -> usize,
        referenced: usize,
    ) {
        let buf_index = self.completed.len();
        let mut compact = ByteBufferMut::with_capacity(referenced);
        self.views_builder.reserve(count);

        let data = bytes.as_slice();
        let mut offset = 0usize;
        for i in 0..count {
            let len = len_at(i);
            let value = &data[offset..offset + len];
            let view = if len > BinaryView::MAX_INLINED_SIZE {
                // In `u32` range: `referenced <= bytes.len() <= MAX_BUFFER_LEN` (checked by the
                // caller), and `compact` never grows past `referenced`.
                #[expect(clippy::cast_possible_truncation)]
                let view = BinaryView::make_view(value, buf_index, compact.len() as u32);
                compact.extend_from_slice(value);
                view
            } else {
                BinaryView::make_view(value, buf_index, 0)
            };
            self.views_builder.push(view);
            offset += len;
        }
        assert_eq!(
            offset,
            data.len(),
            "value lengths must describe the byte heap exactly"
        );

        if !compact.is_empty() {
            let pushed_index = self.completed.push(compact.freeze());
            assert_eq!(pushed_index, buf_index, "Buffer already exists");
        }

        self.push_only_validity_mask(validity);
        debug_assert_eq!(self.nulls.len(), self.views_builder.len());
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
    fn push_only_validity_mask(&mut self, validity_mask: &Mask) {
        self.nulls.append_validity_mask(validity_mask);
    }

    pub(crate) fn append_varbinview_array(
        &mut self,
        array: &VarBinViewArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.flush_in_progress();

        let mask = array.varbinview_validity().execute_mask(array.len(), ctx)?;

        self.push_only_validity_mask(&mask);

        let view_adjustment =
            self.completed
                .extend_from_compaction(BuffersWithOffsets::from_array(
                    array,
                    self.compaction_threshold,
                    ctx,
                ));

        match view_adjustment {
            ViewAdjustment::Precomputed(adjustment) => self.views_builder.extend_trusted(
                array
                    .views()
                    .iter()
                    .map(|view| adjustment.adjust_view(view)),
            ),
            ViewAdjustment::Rewriting(adjustment) => match mask {
                Mask::AllTrue(_) => {
                    for (idx, &view) in array.views().iter().enumerate() {
                        let new_view = self.push_view(view, &adjustment, array, idx);
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
                            self.push_view(view, &adjustment, array, idx)
                        };
                        self.views_builder.push(new_view);
                    }
                }
            },
        }

        Ok(())
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

    fn reserve_exact(&mut self, additional: usize) {
        self.views_builder.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::from_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_varbinview().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::VarBinView(self.finish_into_varbinview())
    }
}

impl VarBinViewBuilder {
    #[inline]
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

    /// Push a new block, returning the index it landed at (or, when deduplicating, the index of
    /// the identical block already held).
    fn push(&mut self, block: ByteBuffer) -> u32 {
        match self {
            Self::Default(buffers) => {
                assert!(buffers.len() < u32::MAX as usize, "Too many blocks");
                buffers.push(block);
                self.len() - 1
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
    pub fn from_array(
        array: &VarBinViewArray,
        compaction_threshold: f64,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        if compaction_threshold == 0.0 {
            return Self::AllKept {
                buffers: Arc::from(
                    array
                        .data_buffers()
                        .iter()
                        .cloned()
                        .map(|b| b.unwrap_host())
                        .collect_vec(),
                ),
                offsets: None,
            };
        }

        let buffer_utilizations = array
            .buffer_utilizations(ctx)
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
    #[inline]
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
    #[inline]
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
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::VarBinViewBuilder;
    use crate::builders::varbinview::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    /// A long-enough value that a view over it must reference a data buffer.
    const LONG: &str = "a value that is far too long to inline";

    /// The heap is adopted zero-copy as a data buffer, the views are built against it in place,
    /// and the append composes with staged in-progress bytes on either side.
    #[test]
    fn test_append_buffer_with_lengths() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 8);

        // Stages an in-progress buffer the bulk append has to flush first.
        builder.append_value(LONG);

        let heap = ByteBuffer::copy_from([LONG.as_bytes(), b"", b"tiny"].concat());
        let heap_ptr = heap.as_ptr();
        let lengths = [u32::try_from(LONG.len()).unwrap(), 0, 4];
        builder.append_buffer_with_lengths(heap, &lengths, &Mask::from_iter([true, false, true]));

        builder.append_value("tail");

        let actual = builder.finish_into_varbinview();
        // The adopted heap sits after the flushed in-progress buffer, untouched.
        assert_eq!(actual.data_buffers()[1].as_host().as_ptr(), heap_ptr);

        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some(LONG),
            Some(LONG),
            None,
            Some("tiny"),
            Some("tail"),
        ]);
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    /// Offsets need not start at zero: only the referenced range of the heap is adopted.
    #[test]
    fn test_append_buffer_with_offsets() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 8);

        let heap = ByteBuffer::copy_from(format!("..{LONG}tiny!!"));
        let long_len = u32::try_from(LONG.len()).unwrap();
        let offsets = [2u32, 2 + long_len, 2 + long_len, 2 + long_len + 4];
        builder.append_buffer_with_offsets(
            heap.clone(),
            &offsets,
            &Mask::from_iter([true, false, true]),
        );

        let actual = builder.finish_into_varbinview();
        // Zero-copy adoption of just the `offsets[0]..offsets[last]` range.
        // SAFETY: offset 2 is in bounds of the heap.
        assert_eq!(actual.data_buffers()[0].as_host().as_ptr(), unsafe {
            heap.as_ptr().add(2)
        });
        assert_eq!(
            actual.data_buffers()[0].len(),
            LONG.len() + 4,
            "only the referenced range must be adopted"
        );

        let expected =
            <VarBinViewArray as FromIterator<_>>::from_iter([Some(LONG), None, Some("tiny")]);
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    /// A compacting builder must measure the heap it is handed: values short enough to inline
    /// never reference it, so an under-utilized heap is rewritten to just the referencing values.
    #[test]
    fn test_append_buffer_with_lengths_compacts_underutilized_heap() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder =
            VarBinViewBuilder::with_compaction(DType::Utf8(Nullability::Nullable), 4, 1.0);

        let heap = ByteBuffer::copy_from([b"short".as_slice(), LONG.as_bytes(), b"tiny"].concat());
        let lengths = [5u32, u32::try_from(LONG.len()).unwrap(), 4];
        builder.append_buffer_with_lengths(heap, &lengths, &Mask::new_true(3));

        let actual = builder.finish_into_varbinview();
        assert_eq!(actual.data_buffers().len(), 1);
        assert_eq!(
            actual.data_buffers()[0].len(),
            LONG.len(),
            "the compact buffer must hold only the non-inlined values"
        );

        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("short"),
            Some(LONG),
            Some("tiny"),
        ]);
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    /// A fully-inlined heap has zero utilization; a compacting builder must not retain it at all.
    #[test]
    fn test_append_buffer_with_lengths_drops_fully_inlined_heap() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder =
            VarBinViewBuilder::with_compaction(DType::Utf8(Nullability::Nullable), 4, 1.0);

        let heap = ByteBuffer::copy_from(b"shorttinysmall".as_slice());
        builder.append_buffer_with_lengths(heap, &[5u32, 4, 5], &Mask::new_true(3));

        let actual = builder.finish_into_varbinview();
        assert!(
            actual.data_buffers().is_empty(),
            "a fully-inlined append must not retain any value bytes"
        );

        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("short"),
            Some("tiny"),
            Some("small"),
        ]);
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    #[test]
    fn test_utf8_builder() {
        let mut ctx = array_session().create_execution_ctx();
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
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    #[test]
    fn test_utf8_builder_with_extend() {
        let mut ctx = array_session().create_execution_ctx();
        let array = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_null();
            builder.append_value("Hello2");
            builder.finish()
        };
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        builder.append_value("Hello1");
        array.append_to_builder(&mut builder, &mut ctx).unwrap();
        builder.append_nulls(2);
        builder.append_value("Hello3");

        let actual = builder.finish_into_canonical(&mut ctx);
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("Hello1"),
            None,
            Some("Hello2"),
            None,
            None,
            Some("Hello3"),
        ]);
        assert_arrays_eq!(actual.into_array(), expected.into_array(), &mut ctx);
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

        let mut ctx = array_session().create_execution_ctx();

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
        let mut ctx = array_session().create_execution_ctx();
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
        assert_arrays_eq!(&array, &expected, &mut ctx);

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
        assert_arrays_eq!(&binary_array, &expected, &mut ctx);

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
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_binary()
            .value()
            .cloned()
            .unwrap();
        assert_eq!(retrieved.len(), 8192);
        assert_eq!(retrieved.as_slice(), &large_value);
    }
}
