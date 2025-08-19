// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_buffer::{Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure};
use vortex_mask::Mask;
use vortex_scalar::{BinaryScalar, Scalar, Utf8Scalar};
use vortex_utils::aliases::hash_map::{Entry, HashMap};

use crate::arrays::{BinaryView, VarBinViewArray};
use crate::builders::{ArrayBuilder, LazyNullBufferBuilder};
use crate::canonical::{Canonical, ToCanonical};
use crate::{Array, ArrayRef, IntoArray};

/// The builder for building a [`VarBinViewArray`].
pub struct VarBinViewBuilder {
    dtype: DType,
    views_builder: BufferMut<BinaryView>,
    nulls: LazyNullBufferBuilder,
    completed: CompletedBuffers,
    in_progress: ByteBufferMut,
    growth_strategy: BufferGrowthStrategy,
}

impl VarBinViewBuilder {
    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        Self::new(dtype, capacity, Default::default(), Default::default())
    }

    pub fn with_buffer_deduplication(dtype: DType, capacity: usize) -> Self {
        Self::new(
            dtype,
            capacity,
            CompletedBuffers::Deduplicated(Default::default()),
            Default::default(),
        )
    }

    pub fn new(
        dtype: DType,
        capacity: usize,
        completed: CompletedBuffers,
        growth_strategy: BufferGrowthStrategy,
    ) -> Self {
        assert!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "VarBinViewBuilder DType must be Utf8 or Binary."
        );
        Self {
            views_builder: BufferMut::<BinaryView>::with_capacity(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            completed,
            in_progress: ByteBufferMut::empty(),
            dtype,
            growth_strategy,
        }
    }

    fn append_value_view(&mut self, value: &[u8]) {
        let length =
            u32::try_from(value.len()).vortex_expect("cannot have a single string >2^32 in length");
        if length <= 12 {
            self.views_builder.push(BinaryView::make_view(value, 0, 0));
            return;
        }

        let required_cap = self.in_progress.len() + value.len();
        if self.in_progress.capacity() < required_cap {
            self.flush_in_progress();
            let next_buffer_size = self.growth_strategy.next_size() as usize;
            let to_reserve = next_buffer_size.max(value.len());
            self.in_progress.reserve(to_reserve);
        };

        let offset = u32::try_from(self.in_progress.len()).vortex_expect("too many buffers");
        self.in_progress.extend_from_slice(value);
        let view = BinaryView::make_view(
            value,
            // buffer offset
            self.completed.len(),
            offset,
        );
        self.views_builder.push(view);
    }

    /// Appends a value to the builder.
    pub fn append_value<S: AsRef<[u8]>>(&mut self, value: S) {
        self.append_value_view(value.as_ref());
        self.nulls.append_non_null();
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

    pub fn completed_block_count(&self) -> u32 {
        self.completed.len()
    }

    // Pushes an array of values into the buffer, where the buffers are sections of a
    // VarBinView and the views are the BinaryView's of the VarBinView *already with their*
    // buffers adjusted.
    // The views must all point to sections of the buffers and the validity length must match
    // the view length.
    /// ## Panics
    /// Panics if this builder deduplicates buffers and if any of the given buffers already
    /// exists on this builder
    pub fn push_buffer_and_adjusted_views(
        &mut self,
        buffer: &[ByteBuffer],
        views: &Buffer<BinaryView>,
        validity_mask: Mask,
    ) {
        self.flush_in_progress();

        let expected_completed_len = self.completed.len() as usize + buffer.len();
        self.completed.extend_from_slice(buffer);
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
}

impl VarBinViewBuilder {
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
            "VarBinViewBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        match self.dtype() {
            DType::Utf8(_) => {
                let utf8_scalar = Utf8Scalar::try_from(scalar)?;
                match utf8_scalar.value() {
                    Some(value) => self.append_value(value),
                    None => self.append_null(),
                }
            }
            DType::Binary(_) => {
                let binary_scalar = BinaryScalar::try_from(scalar)?;
                match binary_scalar.value() {
                    Some(value) => self.append_value(value),
                    None => self.append_null(),
                }
            }
            _ => vortex_bail!(
                "VarBinViewBuilder can only handle Utf8 or Binary scalars, got {:?}",
                scalar.dtype()
            ),
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let array = array.to_varbinview();
        self.flush_in_progress();

        let new_indices = self.completed.extend_from_slice(array.buffers());

        match new_indices {
            NewIndices::ConstantOffset(offset) => {
                self.views_builder
                    .extend_trusted(array.views().iter().map(|view| view.offset_view(offset)));
            }
            NewIndices::LookupArray(lookup) => {
                self.views_builder
                    .extend_trusted(array.views().iter().map(|view| {
                        if view.is_inlined() {
                            *view
                        } else {
                            let new_buffer_idx = lookup[view.as_view().buffer_index() as usize];
                            view.with_buffer_idx(new_buffer_idx)
                        }
                    }));
            }
        }

        self.push_only_validity_mask(array.validity_mask());
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.views_builder.capacity() {
            self.views_builder
                .reserve(capacity - self.views_builder.len());
            self.nulls.ensure_capacity(capacity);
        }
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_varbinview().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::VarBinView(self.finish_into_varbinview())
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
#[allow(clippy::cast_possible_truncation)]
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

    fn extend_from_slice(&mut self, new_buffers: &[ByteBuffer]) -> NewIndices {
        match self {
            Self::Default(buffers) => {
                let offset = buffers.len() as u32;
                buffers.extend_from_slice(new_buffers);
                NewIndices::ConstantOffset(offset)
            }
            Self::Deduplicated(buffers) => {
                NewIndices::LookupArray(buffers.extend_from_slice(new_buffers))
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

enum NewIndices {
    // add a constant offset to get the new idx
    ConstantOffset(u32),
    // lookup from the given array to get the new idx
    LookupArray(Vec<u32>),
}

#[derive(Default)]
pub struct DeduplicatedBuffers {
    buffers: Vec<ByteBuffer>,
    buffer_to_idx: HashMap<BufferId, u32>,
}

impl DeduplicatedBuffers {
    // Self::push enforces len < u32::max
    #[allow(clippy::cast_possible_truncation)]
    fn len(&self) -> u32 {
        self.buffers.len() as u32
    }

    /// Push a new block if not seen before. Returns the idx of the block.
    fn push(&mut self, block: ByteBuffer) -> u32 {
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

    fn extend_from_slice(&mut self, buffers: &[ByteBuffer]) -> Vec<u32> {
        buffers
            .iter()
            .map(|buffer| self.push(buffer.clone()))
            .collect()
    }

    fn finish(self) -> Arc<[ByteBuffer]> {
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

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use itertools::Itertools;
    use vortex_dtype::{DType, Nullability};

    use crate::accessor::ArrayAccessor;
    use crate::arrays::VarBinViewVTable;
    use crate::builders::{ArrayBuilder, VarBinViewBuilder};

    #[test]
    fn test_utf8_builder() {
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        builder.append_value("Hello");
        builder.append_null();
        builder.append_value("World");

        builder.append_nulls(2);

        builder.append_zeros(2);
        builder.append_value("test");

        let arr = builder.finish();

        let arr = arr
            .as_::<VarBinViewVTable>()
            .with_iterator(|iter| {
                iter.map(|x| x.map(|x| from_utf8(x).unwrap().to_string()))
                    .collect_vec()
            })
            .unwrap();
        assert_eq!(arr.len(), 8);
        assert_eq!(
            arr,
            vec![
                Some("Hello".to_string()),
                None,
                Some("World".to_string()),
                None,
                None,
                Some("".to_string()),
                Some("".to_string()),
                Some("test".to_string()),
            ]
        );
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

        let arr = builder.finish_into_canonical().into_varbinview();

        let arr = arr
            .with_iterator(|iter| {
                iter.map(|x| x.map(|x| from_utf8(x).unwrap().to_string()))
                    .collect_vec()
            })
            .unwrap();
        assert_eq!(arr.len(), 6);
        assert_eq!(
            arr,
            vec![
                Some("Hello1".to_string()),
                None,
                Some("Hello2".to_string()),
                None,
                None,
                Some("Hello3".to_string()),
            ]
        );
    }

    #[test]
    fn test_buffer_deduplication() {
        let array = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_value("This is a long string that should not be inlined");
            builder.append_value("short string");
            builder.finish_into_varbinview()
        };

        assert_eq!(array.buffers().len(), 1);
        let mut builder =
            VarBinViewBuilder::with_buffer_deduplication(DType::Utf8(Nullability::Nullable), 10);

        array.append_to_builder(&mut builder);
        assert_eq!(builder.completed_block_count(), 1);

        array.slice(1..2).append_to_builder(&mut builder);
        array.slice(0..1).append_to_builder(&mut builder);
        assert_eq!(builder.completed_block_count(), 1);

        let array2 = {
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);
            builder.append_value("This is a long string that should not be inlined");
            builder.finish_into_varbinview()
        };

        array2.append_to_builder(&mut builder);
        assert_eq!(builder.completed_block_count(), 2);

        array.slice(0..1).append_to_builder(&mut builder);
        array2.slice(0..1).append_to_builder(&mut builder);
        assert_eq!(builder.completed_block_count(), 2);
    }

    #[test]
    fn test_append_scalar() {
        use vortex_scalar::Scalar;

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
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.
        use crate::array::Array;
        let scalar0 = array.scalar_at(0).as_utf8().value();
        assert_eq!(scalar0.as_ref().map(|s| s.as_str()), Some("hello"));

        let scalar1 = array.scalar_at(1).as_utf8().value();
        assert_eq!(scalar1.as_ref().map(|s| s.as_str()), Some("world"));

        let scalar2 = array.scalar_at(2).as_utf8().value();
        assert_eq!(scalar2, None); // This should be null.

        // Test with Binary builder.
        let mut binary_builder =
            VarBinViewBuilder::with_capacity(DType::Binary(Nullability::Nullable), 10);

        let binary_scalar = Scalar::binary(vec![1u8, 2, 3], Nullability::Nullable);
        binary_builder.append_scalar(&binary_scalar).unwrap();

        let binary_null = Scalar::null(DType::Binary(Nullability::Nullable));
        binary_builder.append_scalar(&binary_null).unwrap();

        let binary_array = binary_builder.finish();
        assert_eq!(binary_array.len(), 2);

        // Check actual binary values.
        let binary0 = binary_array.scalar_at(0).as_binary().value();
        assert_eq!(
            binary0.as_ref().map(|b| b.as_slice()),
            Some(&[1u8, 2, 3][..])
        );

        let binary1 = binary_array.scalar_at(1).as_binary().value();
        assert_eq!(binary1, None); // This should be null.

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
        use super::{BufferGrowthStrategy, VarBinViewBuilder};

        let mut builder = VarBinViewBuilder::new(
            DType::Binary(Nullability::Nullable),
            10,
            Default::default(),
            BufferGrowthStrategy::exponential(1024, 4096),
        );

        // Create a value larger than max_size
        let large_value = vec![0u8; 8192];

        // Should successfully append the large value
        builder.append_value(&large_value);

        let array = builder.finish_into_varbinview();
        assert_eq!(array.len(), 1);

        // Verify the value was stored correctly
        let retrieved = array.scalar_at(0).as_binary().value().unwrap();
        assert_eq!(retrieved.len(), 8192);
        assert_eq!(retrieved.as_slice(), &large_value);
    }
}
