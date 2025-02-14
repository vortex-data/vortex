use std::any::Any;
use std::cmp::max;

use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::array::{BinaryView, VarBinViewArray};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::ArrayBuilder;
use crate::{Array, Canonical, IntoArray, IntoCanonical};

pub struct VarBinViewBuilder {
    views_builder: BufferMut<BinaryView>,
    null_buffer_builder: LazyNullBufferBuilder,
    completed: Vec<ByteBuffer>,
    in_progress: ByteBufferMut,
    nullability: Nullability,
    dtype: DType,
}

impl VarBinViewBuilder {
    // TODO(joe): add a block growth strategy, from arrow
    const BLOCK_SIZE: u32 = 8 * 8 * 1024;

    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        Self {
            views_builder: BufferMut::<BinaryView>::with_capacity(capacity),
            null_buffer_builder: LazyNullBufferBuilder::new(capacity),
            completed: vec![],
            in_progress: ByteBufferMut::with_capacity(VarBinViewBuilder::BLOCK_SIZE as usize),
            nullability: dtype.nullability(),
            dtype,
        }
    }

    fn append_value_view(&mut self, value: &[u8]) {
        let v: &[u8] = value;
        let length =
            u32::try_from(v.len()).vortex_expect("cannot have a single string >2^32 in length");
        if length <= 12 {
            self.views_builder.push(BinaryView::new_inlined(v));
            return;
        }

        let required_cap = self.in_progress.len() + v.len();
        if self.in_progress.capacity() < required_cap {
            self.flush_in_progress();
            let to_reserve = max(v.len(), VarBinViewBuilder::BLOCK_SIZE as usize);
            self.in_progress.reserve(to_reserve);
        };
        let offset = u32::try_from(self.in_progress.len()).vortex_expect("too many buffers");
        self.in_progress.extend_from_slice(v);

        let view = BinaryView::new_view(
            length,
            // inline the first 4 bytes of the view
            v[0..4].try_into().vortex_expect("length already checked"),
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
            let f = ByteBuffer::from(std::mem::take(&mut self.in_progress).freeze());
            self.push_completed(f)
        }
    }

    fn push_completed(&mut self, block: ByteBuffer) {
        assert!(block.len() < u32::MAX as usize, "Block too large");
        assert!(self.completed.len() < u32::MAX as usize, "Too many blocks");
        self.completed.push(block);
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
    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = if let Some(array) = VarBinViewArray::maybe_from(&array) {
            array
        } else {
            let Ok(Canonical::VarBinView(array)) = array.clone().into_canonical() else {
                vortex_bail!("Expected Canonical::VarBinView, found {:?}", array);
            };
            array
        };

        self.flush_in_progress();

        let buffers_offset = u32::try_from(self.completed.len())?;
        self.completed.extend(array.buffers());

        self.views_builder
            .extend(array.views().into_iter().map(|view| {
                if view.is_inlined() {
                    view
                } else {
                    // Referencing views must have their buffer_index adjusted with new offsets
                    let view_ref = view.as_view();
                    BinaryView::new_view(
                        view.len(),
                        *view_ref.prefix(),
                        buffers_offset + view_ref.buffer_index(),
                        view_ref.offset(),
                    )
                }
            }));

        self.null_buffer_builder
            .append_validity_mask(array.validity_mask()?);

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        self.flush_in_progress();
        let buffers = std::mem::take(&mut self.completed);

        let validity = self
            .null_buffer_builder
            .finish_with_nullability(self.nullability)?;

        Ok(VarBinViewArray::try_new(
            std::mem::take(&mut self.views_builder).freeze(),
            buffers,
            std::mem::replace(&mut self.dtype, DType::Null),
            validity,
        )?
        .into_array())
    }
}
