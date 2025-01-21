use std::any::Any;
use std::marker::PhantomData;
use std::mem::take;

use arrow_buffer::NullBufferBuilder;
use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::array::{BinaryView, BoolArray, VarBinViewArray};
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData};

pub trait ViewDType: 'static + Send + Sync {
    type RefType: ?Sized;

    fn as_bytes(value: &Self::RefType) -> &[u8];

    fn dtype(nullability: Nullability) -> DType;
}

pub struct UTF8DType;
impl ViewDType for UTF8DType {
    type RefType = str;

    fn as_bytes(value: &Self::RefType) -> &[u8] {
        value.as_bytes()
    }

    fn dtype(nullability: Nullability) -> DType {
        DType::Utf8(nullability)
    }
}

pub struct BinaryDType;
impl ViewDType for BinaryDType {
    type RefType = [u8];

    fn as_bytes(value: &Self::RefType) -> &[u8] {
        value
    }

    fn dtype(nullability: Nullability) -> DType {
        DType::Binary(nullability)
    }
}

pub type Utf8Builder = ViewBuilder<UTF8DType>;
pub type BinaryBuilder = ViewBuilder<BinaryDType>;

pub struct ViewBuilder<V: ViewDType> {
    views: BufferMut<BinaryView>,
    validity: NullBufferBuilder,
    completed: Vec<ByteBuffer>,
    in_progress: ByteBufferMut,
    dtype: DType,
    _marker: PhantomData<V>,
}

impl<V: ViewDType> ViewBuilder<V> {
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            views: BufferMut::with_capacity(capacity),
            validity: NullBufferBuilder::new(capacity),
            completed: vec![],
            in_progress: ByteBufferMut::empty(),
            dtype: V::dtype(nullability),
            _marker: PhantomData,
        }
    }

    /// Append a new buffer to the builder, flushing any in-progress buffer.
    /// Returns the buffer index that should be referenced in the views.
    pub fn append_buffer(&mut self, buffer: ByteBuffer) -> u32 {
        assert!(buffer.len() <= u32::MAX as usize);
        let idx = self.completed.len();
        self.completed.push(buffer);
        idx as u32
    }

    /// Append a value to the builder.
    pub fn append_value<S: AsRef<V::RefType>>(&mut self, value: S) {
        let view = self.create_view(value.as_ref());
        self.views.push(view);
        self.validity.append_non_null();
    }

    pub fn append_values<S: AsRef<V::RefType>>(&mut self, value: S, n: usize) {
        let view = self.create_view(value.as_ref());
        self.views.push_n(view, n);
        self.validity.append_n_non_nulls(n);
    }

    pub fn append_option<S: AsRef<V::RefType>>(&mut self, value: Option<S>) {
        match value {
            None => self.append_null(),
            Some(value) => self.append_value(value),
        }
    }

    /// Push a value onto the in-progress buffer, returning its buffer idx and offset.
    fn create_view(&mut self, value: &V::RefType) -> BinaryView {
        let value = V::as_bytes(value);
        let len = u32::try_from(value.len()).vortex_expect("Value length must be <= u32");
        if len <= 12 {
            BinaryView::new_inlined(value)
        } else {
            // Flush the in-progress buffer if we're going to overflow
            // TODO(ngates): we could use different strategies here, e.g. smaller buffers.
            if self.in_progress.len() + len as usize > u32::MAX as usize {
                self.flush();
            }

            let offset =
                u32::try_from(self.in_progress.len()).vortex_expect("Buffer length must be <= u32");
            self.in_progress.extend_from_slice(value);

            let mut prefix = [0; 4];
            prefix.copy_from_slice(&value[0..4]);

            BinaryView::new_view(len, prefix, self.current_buffer_idx(), offset)
        }
    }

    /// Append a view to the builder, without checking that the buffer idx or offsets are correct.
    pub unsafe fn push_view_unchecked(&mut self, view: BinaryView) {
        self.views.push(view);
        self.validity.append_non_null();
    }

    pub fn current_buffer_idx(&self) -> u32 {
        self.completed
            .len()
            .try_into()
            .vortex_expect("Buffer index must be <= u32")
    }

    /// Flush any in-progress buffer to the completed buffers.
    fn flush(&mut self) {
        if !self.in_progress.is_empty() {
            let buffer = take(&mut self.in_progress).freeze();
            self.completed.push(buffer);
        }
    }
}

impl<V: ViewDType> ArrayBuilder for ViewBuilder<V> {
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
        self.views.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.views.push_n(BinaryView::new_inlined(&[]), n);
        self.validity.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.views.push_n(BinaryView::new_inlined(&[]), n);
        self.validity.append_n_nulls(n);
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        self.flush();
        assert!(self.in_progress.is_empty());

        let validity = match (self.validity.finish(), self.dtype().nullability()) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (Some(_), Nullability::NonNullable) => {
                vortex_bail!("Non-nullable builder has null values")
            }
            (None, Nullability::Nullable) => Validity::AllValid,
            (Some(nulls), Nullability::Nullable) => {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::Array(BoolArray::from(nulls.into_inner()).into_array())
                }
            }
        };

        Ok(VarBinViewArray::try_new(
            take(&mut self.views).freeze(),
            take(&mut self.completed),
            self.dtype().clone(),
            validity,
        )?
        .into_array())
    }
}
