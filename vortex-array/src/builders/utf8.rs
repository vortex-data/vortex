use std::any::Any;
use std::cmp::max;

use vortex_buffer::{BufferMut, ByteBuffer};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect, VortexResult};
use vortex_mask::AllOr;

use crate::array::{BinaryView, VarBinViewArray};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::{Array, Canonical, IntoArray, IntoCanonical};

pub struct Utf8Builder {
    views_builder: BufferMut<BinaryView>,
    null_buffer_builder: LazyNullBufferBuilder,
    completed: Vec<ByteBuffer>,
    in_progress: Vec<u8>,
    nullability: Nullability,
    dtype: DType,
}

impl Utf8Builder {
    // TODO(joe): add a block growth strategy, from arrow
    const BLOCK_SIZE: u32 = 8 * 8 * 1024;

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            views_builder: BufferMut::<BinaryView>::with_capacity(capacity),
            null_buffer_builder: LazyNullBufferBuilder::new(capacity),
            completed: vec![],
            in_progress: vec![],
            nullability,
            dtype: DType::Utf8(nullability),
        }
    }

    fn append_value_view(&mut self, value: &str) {
        let v: &[u8] = value.as_ref();
        let length =
            u32::try_from(v.len()).vortex_expect("cannot have a single string >2^32 in length");
        if length <= 12 {
            self.views_builder.push(BinaryView::new_inlined(v));
            return;
        }

        let required_cap = self.in_progress.len() + v.len();
        if self.in_progress.capacity() < required_cap {
            self.flush_in_progress();
            let to_reserve = max(v.len(), Utf8Builder::BLOCK_SIZE as usize);
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
    pub fn append_value<S: AsRef<str>>(&mut self, value: S) {
        self.append_value_view(value.as_ref());
        self.null_buffer_builder.append_non_null();
    }

    #[inline]
    pub fn append_option<S: AsRef<str>>(&mut self, value: Option<S>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.null_buffer_builder.append_null();
        self.views_builder.push(BinaryView::empty_view());
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

    pub fn finish2(mut self) -> VortexResult<Array> {
        self.flush_in_progress();
        let buffers = std::mem::take(&mut self.completed);

        let nulls = self.null_buffer_builder.finish();
        let validity = match (self.nullability, nulls) {
            (NonNullable, None) => Validity::NonNullable,
            (Nullable, None) => Validity::AllValid,
            (Nullable, Some(arr)) => Validity::from(arr),
            _ => vortex_panic!("Invalid nullability/nulls combination"),
        };

        Ok(VarBinViewArray::try_new(
            // TODO(joe): remove clone.
            self.views_builder.freeze(),
            buffers,
            // TODO(joe): remove clone.
            self.dtype.clone(),
            validity,
        )?
        .into_array())
    }
}

impl ArrayBuilder for Utf8Builder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

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

        // array

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

        match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                self.null_buffer_builder.append_n_non_nulls(array.len());
            }
            AllOr::None => self.null_buffer_builder.append_n_nulls(array.len()),
            AllOr::Some(validity) => self.null_buffer_builder.append_buffer(validity.clone()),
        }

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        self.flush_in_progress();
        let buffers = std::mem::take(&mut self.completed);

        let nulls = self.null_buffer_builder.finish();
        let validity = match (self.nullability, nulls) {
            (NonNullable, None) => Validity::NonNullable,
            (Nullable, None) => Validity::AllValid,
            (Nullable, Some(arr)) => Validity::from(arr),
            _ => vortex_panic!("Invalid nullability/nulls combination"),
        };

        Ok(VarBinViewArray::try_new(
            // TODO(joe): remove clone.
            self.views_builder.clone().freeze(),
            buffers,
            // TODO(joe): remove clone.
            self.dtype.clone(),
            validity,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use itertools::Itertools;
    use vortex_dtype::Nullability;

    use crate::accessor::ArrayAccessor;
    use crate::array::VarBinViewArray;
    use crate::builders::{ArrayBuilder, Utf8Builder};

    #[test]
    fn test_utf8_builder() {
        let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);

        builder.append_option(Some("Hello"));
        builder.append_option::<&str>(None);
        builder.append_value("World");

        builder.append_nulls(2);

        builder.append_zeros(2);
        builder.append_value("test");

        let arr = VarBinViewArray::try_from(builder.finish().unwrap()).unwrap();

        let arr = arr
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
            let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);
            builder.append_null();
            builder.append_value("Hello2");
            builder.finish().unwrap()
        };
        let mut builder = Utf8Builder::with_capacity(Nullability::Nullable, 10);

        builder.append_option(Some("Hello1"));
        builder.extend_from_array(array).unwrap();
        builder.append_nulls(2);
        builder.append_value("Hello3");

        let arr = VarBinViewArray::try_from(builder.finish().unwrap()).unwrap();

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
