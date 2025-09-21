// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::cmp::max;
use std::mem::size_of;
use std::sync::Arc;

use vortex_buffer::{Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure};
use vortex_mask::Mask;
use vortex_scalar::{BinaryScalar, Scalar, Utf8Scalar};
use vortex_utils::aliases::hash_map::{Entry, HashMap};

use crate::arrays::{BinaryView, VarBinViewArray};
use crate::builders::{ArrayBuilder, ExtendResult, LazyNullBufferBuilder};
use crate::canonical::{Canonical, ToCanonical};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray};

/// The builder for building a [`VarBinViewArray`].
pub struct VarBinViewBuilder {
    dtype: DType,
    views_builder: BufferMut<BinaryView>,
    nulls: LazyNullBufferBuilder,
    completed: CompletedBuffers,
    in_progress: ByteBufferMut,
    size_limit: usize,
    current_nbytes: usize,
}

impl VarBinViewBuilder {
    // TODO(joe): add a block growth strategy, from arrow
    const BLOCK_SIZE: u32 = 8 * 8 * 1024;

    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        Self::new(dtype, capacity, Default::default(), usize::MAX)
    }

    pub fn with_capacity_and_limit(dtype: DType, capacity: usize, size_limit: usize) -> Self {
        Self::new(dtype, capacity, Default::default(), size_limit)
    }

    pub fn with_buffer_deduplication(dtype: DType, capacity: usize) -> Self {
        Self::new(
            dtype,
            capacity,
            CompletedBuffers::Deduplicated(Default::default()),
            usize::MAX,
        )
    }

    pub fn with_buffer_deduplication_and_limit(dtype: DType, capacity: usize, size_limit: usize) -> Self {
        Self::new(
            dtype,
            capacity,
            CompletedBuffers::Deduplicated(Default::default()),
            size_limit,
        )
    }

    fn new(dtype: DType, capacity: usize, completed: CompletedBuffers, size_limit: usize) -> Self {
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
            size_limit,
            current_nbytes: 0,
        }
    }

    fn append_value_view(&mut self, value: &[u8]) {
        let length =
            u32::try_from(value.len()).vortex_expect("cannot have a single string >2^32 in length");

        let bytes_added = if length <= 12 {
            // Inlined view - only adds view size
            self.views_builder.push(BinaryView::make_view(value, 0, 0));
            size_of::<BinaryView>() + 1 / 8 // view + validity bit
        } else {
            // Non-inlined view - adds view + data
            let required_cap = self.in_progress.len() + value.len();
            if self.in_progress.capacity() < required_cap {
                self.flush_in_progress();
                let to_reserve = max(value.len(), VarBinViewBuilder::BLOCK_SIZE as usize);
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
            size_of::<BinaryView>() + value.len() + 1 / 8 // view + data + validity bit
        };

        self.update_size(bytes_added);
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

    /// Calculates the estimated size of adding a single element at the given index.
    fn estimate_element_size(&self, array: &VarBinViewArray, index: usize) -> usize {
        let view = &array.views()[index];
        let view_size = size_of::<BinaryView>();
        let validity_size = 1; // 1 bit, rounded up

        if view.is_inlined() {
            // Inlined views don't add buffer data
            view_size + validity_size / 8
        } else {
            // Non-inlined views add buffer data
            let data_size = view.len() as usize;
            view_size + data_size + validity_size / 8
        }
    }

    /// Updates the cached size when adding elements.
    fn update_size(&mut self, bytes_added: usize) {
        self.current_nbytes += bytes_added;
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
    ) -> VortexResult<ExtendResult> {
        // Calculate the estimated size this operation would add
        let estimated_bytes = buffer.iter().map(|buf| buf.len()).sum::<usize>()
            + views.len() * size_of::<BinaryView>()
            + validity_mask.len() / 8; // Approximate validity size

        // Check if we have space for the entire operation
        if self.current_nbytes + estimated_bytes > self.size_limit {
            return Ok(ExtendResult::empty());
        }

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

        debug_assert_eq!(self.nulls.len(), self.views_builder.len());

        // Update cached size
        self.update_size(estimated_bytes);

        Ok(ExtendResult::complete(estimated_bytes, views.len()))
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
                std::mem::replace(&mut self.dtype, DType::Null),
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

    fn extend_from_array_element_by_element(&mut self, array: &VarBinViewArray) -> VortexResult<ExtendResult> {
        let mut elements_consumed = 0;
        let mut bytes_consumed = 0;

        // Process elements one by one until we hit the size limit
        for i in 0..array.len() {
            let element_size = self.estimate_element_size(array, i);

            // Check if adding this element would exceed the size limit
            if self.current_nbytes + element_size > self.size_limit {
                break; // Stop here, return partial result
            }

            // Add this element
            let view = &array.views()[i];

            // Handle validity for both inlined and non-inlined cases
            if array.validity().is_valid(i) {
                if view.is_inlined() {
                    // Inlined view - just add the view
                    self.views_builder.push(*view);
                    self.nulls.append_non_null();
                } else {
                    // Non-inlined view - need to copy buffer data
                    let view_ref = view.as_view();
                    let buffer_idx = view_ref.buffer_index() as usize;
                    let source_buffer = &array.buffers()[buffer_idx];
                    let range = view_ref.to_range();
                    let data = &source_buffer.as_slice()[range];

                    // Add the data using our append_value logic
                    self.append_value(data);
                }
            } else {
                // Null element - use empty view for inlined, or append null
                if view.is_inlined() {
                    self.views_builder.push(*view);  // Keep the inlined view as is
                    self.nulls.append_null();
                } else {
                    self.append_null();
                }
            }

            // Update counters
            bytes_consumed += element_size;
            elements_consumed += 1;
            self.update_size(element_size);
        }

        Ok(ExtendResult::new(bytes_consumed, elements_consumed))
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

    fn nbytes(&self) -> usize {
        self.current_nbytes
    }

    fn append_zeros(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.nulls.append_n_non_nulls(n);
        // Empty views are inlined, so just view + validity
        let bytes_added = n * (size_of::<BinaryView>() + 1 / 8);
        self.update_size(bytes_added);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.nulls.append_n_nulls(n);
        // Null views are inlined empty views, so just view + validity
        let bytes_added = n * (size_of::<BinaryView>() + 1 / 8);
        self.update_size(bytes_added);
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

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) -> VortexResult<ExtendResult> {
        let array = array.to_varbinview();

        // First, try the bulk approach for efficiency and buffer preservation
        let bulk_result = self.push_buffer_and_adjusted_views(
            array.buffers(),
            array.views(),
            array.validity_mask(),
        )?;

        // If bulk approach succeeded (consumed everything), return it
        if bulk_result.elements_consumed == array.len() {
            return Ok(bulk_result);
        }

        // If bulk approach returned empty (no space), fall back to element-by-element
        if bulk_result.is_empty() {
            return self.extend_from_array_element_by_element(&array);
        }

        // If we got a partial result from bulk approach, this shouldn't happen
        // since push_buffer_and_adjusted_views is all-or-nothing, but handle it
        Ok(bulk_result)
    }


    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.views_builder.capacity() {
            self.views_builder
                .reserve(capacity - self.views_builder.len());
            self.nulls.ensure_capacity(capacity);
        }
    }

    fn set_validity(&mut self, validity: Mask) {
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

enum CompletedBuffers {
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

    fn iter(&self) -> impl Iterator<Item = &ByteBuffer> {
        match self {
            Self::Default(buffers) => buffers.iter(),
            Self::Deduplicated(buffers) => buffers.buffers.iter(),
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
struct DeduplicatedBuffers {
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
        let _ = builder.extend_from_array(&array);
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
}
