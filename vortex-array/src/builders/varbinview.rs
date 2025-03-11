use std::any::Any;
use std::cmp::max;

use vortex_buffer::{BufferMut, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{BinaryView, VarBinViewArray};
use crate::builders::ArrayBuilder;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::{Array, ArrayRef, ToCanonical};

pub struct VarBinViewBuilder {
    views_builder: BufferMut<BinaryView>,
    pub null_buffer_builder: LazyNullBufferBuilder,
    completed: Vec<ByteBuffer>,
    in_progress: Vec<u8>,
    nullability: Nullability,
    dtype: DType,
}

impl VarBinViewBuilder {
    // TODO(joe): add a block growth strategy, from arrow
    const BLOCK_SIZE: u32 = 8 * 8 * 1024;

    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        assert!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "VarBinViewBuilder DType must be Utf8 or Binary."
        );
        Self {
            views_builder: BufferMut::<BinaryView>::with_capacity(capacity),
            null_buffer_builder: LazyNullBufferBuilder::new(capacity),
            completed: vec![],
            in_progress: vec![],
            nullability: dtype.nullability(),
            dtype,
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
            let to_reserve = max(value.len(), VarBinViewBuilder::BLOCK_SIZE as usize);
            self.in_progress.reserve(to_reserve);
        };

        let offset = u32::try_from(self.in_progress.len()).vortex_expect("too many buffers");
        self.in_progress.extend_from_slice(value);
        let view = BinaryView::make_view(
            value,
            // buffer offset
            u32::try_from(self.completed.len()).vortex_expect("too many buffers"),
            offset,
        );
        self.views_builder.push(view);
    }

    #[inline]
    pub fn append_value<S: AsRef<[u8]>>(&mut self, value: S) {
        self.append_value_view(value.as_ref());
        self.null_buffer_builder.append_non_null();
    }

    #[inline]
    pub fn append_option<S: AsRef<[u8]>>(&mut self, value: Option<S>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }

    #[inline]
    fn flush_in_progress(&mut self) {
        if !self.in_progress.is_empty() {
            let f = ByteBuffer::from(std::mem::take(&mut self.in_progress));
            self.push_completed(f)
        }
    }

    fn push_completed(&mut self, block: ByteBuffer) {
        assert!(block.len() < u32::MAX as usize, "Block too large");
        assert!(self.completed.len() < u32::MAX as usize, "Too many blocks");
        self.completed.push(block);
    }

    pub fn completed_block_count(&self) -> usize {
        self.completed.len()
    }

    // Pushes an array of values into the buffer, where the buffers are sections of a
    // VarBinView and the views are the BinaryView's of the VarBinView *already with their*
    // buffers adjusted.
    // The views must all point to sections of the buffers and the validity length must match
    // the view length.
    pub fn push_buffer_and_adjusted_views(
        &mut self,
        buffer: impl IntoIterator<Item = ByteBuffer>,
        views: impl IntoIterator<Item = BinaryView>,
        validity_mask: Mask,
    ) {
        self.flush_in_progress();

        self.completed.extend(buffer);
        self.views_builder.extend(views);
        self.push_only_validity_mask(validity_mask);

        debug_assert_eq!(self.null_buffer_builder.len(), self.views_builder.len())
    }

    // Pushes a validity mask into the builder not affecting the views or buffers
    fn push_only_validity_mask(&mut self, validity_mask: Mask) {
        self.null_buffer_builder.append_validity_mask(validity_mask);
    }
}

impl ArrayBuilder for VarBinViewBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    fn len(&self) -> usize {
        self.null_buffer_builder.len()
    }

    #[inline]
    fn append_zeros(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.null_buffer_builder.append_n_non_nulls(n);
    }

    #[inline]
    fn append_nulls(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.null_buffer_builder.append_n_nulls(n);
    }

    #[inline]
    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        let array = array.to_varbinview()?;
        self.flush_in_progress();

        let buffers_offset = u32::try_from(self.completed.len())?;
        self.completed.extend_from_slice(array.buffers());

        self.views_builder.extend(
            array
                .views()
                .iter()
                .map(|view| view.offset_view(buffers_offset)),
        );

        self.push_only_validity_mask(array.validity_mask()?);

        Ok(())
    }

    fn finish(&mut self) -> ArrayRef {
        self.flush_in_progress();
        let buffers = std::mem::take(&mut self.completed);

        assert_eq!(
            self.views_builder.len(),
            self.null_buffer_builder.len(),
            "View and validity length must match"
        );

        let validity = self
            .null_buffer_builder
            .finish_with_nullability(self.nullability);

        VarBinViewArray::try_new(
            std::mem::take(&mut self.views_builder).freeze(),
            buffers,
            std::mem::replace(&mut self.dtype, DType::Null),
            validity,
        )
        .vortex_expect("VarBinViewArray components should be valid.")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use itertools::Itertools;
    use vortex_dtype::{DType, Nullability};

    use crate::ToCanonical;
    use crate::accessor::ArrayAccessor;
    use crate::array::ArrayExt;
    use crate::arrays::VarBinViewArray;
    use crate::builders::{ArrayBuilder, VarBinViewBuilder};

    #[test]
    fn test_utf8_builder() {
        let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 10);

        builder.append_option(Some("Hello"));
        builder.append_option::<&str>(None);
        builder.append_value("World");

        builder.append_nulls(2);

        builder.append_zeros(2);
        builder.append_value("test");

        let arr = builder.finish();

        let arr = arr
            .as_::<VarBinViewArray>()
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

        builder.append_option(Some("Hello1"));
        builder.extend_from_array(&array).unwrap();
        builder.append_nulls(2);
        builder.append_value("Hello3");

        let arr = builder.finish().to_varbinview().unwrap();

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
}
