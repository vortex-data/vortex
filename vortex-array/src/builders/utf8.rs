use std::any::Any;
use std::cmp::max;

use arrow_buffer::{Buffer, NullBufferBuilder};
use vortex_buffer::{Alignment, BufferMut, ByteBuffer};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect, VortexResult};

use crate::array::{BinaryView, VarBinViewArray};
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::{Array, Canonical, IntoArray, IntoCanonical};

// pub struct GenericByteViewBuilder {
//     views_builder: BufferBuilder<u128>,
//     null_buffer_builder: NullBufferBuilder,
//     completed: Vec<Buffer>,
//     in_progress: Vec<u8>,
// }

pub struct Utf8Builder {
    // views_builder: BufferBuilder<u128>,
    views_builder: BufferMut<BinaryView>,

    null_buffer_builder: NullBufferBuilder,
    completed: Vec<Buffer>,
    in_progress: Vec<u8>,
    nullability: Nullability,
    dtype: DType,
}

impl Utf8Builder {
    // TODO(joe): add a block growth strategy, from arrow
    const BLOCK_SIZE: u32 = 8 * 1024;

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            // views_builder: BufferBuilder::new(capacity),
            views_builder: BufferMut::<BinaryView>::empty(),
            null_buffer_builder: NullBufferBuilder::new(capacity),
            completed: vec![],
            in_progress: vec![],
            nullability,
            dtype: DType::Utf8(nullability),
        }
    }

    fn append_value_view(&mut self, value: &str) {
        let v: &[u8] = value.as_ref();
        let length: u32 = v.len().try_into().unwrap();
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
        let offset = self.in_progress.len() as u32;
        self.in_progress.extend_from_slice(v);

        let view = BinaryView::new_view(
            length,
            // inline the first 4 bytes of the view
            v[0..4].try_into().vortex_expect("length already checked"),
            // buffer offset
            self.completed.len() as u32,
            offset,
        );
        self.views_builder.push(view.into());
    }

    pub fn append_value<S: AsRef<str>>(&mut self, value: S) {
        self.append_value_view(value.as_ref());
        self.null_buffer_builder.append_non_null();
    }

    pub fn append_option<S: AsRef<str>>(&mut self, value: Option<S>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }

    pub fn append_null(&mut self) {
        self.null_buffer_builder.append_null();
        self.views_builder.push(BinaryView::empty_view());
    }

    #[inline]
    fn flush_in_progress(&mut self) {
        if !self.in_progress.is_empty() {
            let f = Buffer::from_vec(std::mem::take(&mut self.in_progress));
            self.push_completed(f)
        }
    }

    fn push_completed(&mut self, block: Buffer) {
        assert!(block.len() < u32::MAX as usize, "Block too large");
        assert!(self.completed.len() < u32::MAX as usize, "Too many blocks");
        self.completed.push(block);
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

    fn len(&self) -> usize {
        self.null_buffer_builder.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.null_buffer_builder.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.views_builder.push_n(BinaryView::empty_view(), n);
        self.null_buffer_builder.append_n_nulls(n);
    }

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = if let Some(array) = VarBinViewArray::maybe_from(&array) {
            array
        } else {
            let Ok(Canonical::VarBinView(array)) = array.clone().into_canonical() else {
                vortex_bail!("Expected Canonical::VarBinView, found {:?}", array);
            };
            array
        };

        let _ = array;

        todo!("array {}", array.tree_display());
        // let mut first_offset = None;
        // for buf in array.buffers() {
        //     let offset = self.inner.append_block(buf.into_arrow_buffer());
        //     if first_offset.is_none() {
        //         first_offset = Some(offset);
        //     }
        // }
        // let first_offset = first_offset.unwrap_or(0);
        //
        // for view in array.views().iter() {
        //     if view.is_inlined() {
        //         self.inner.append_value(*view);
        //     } else {
        //         let view_ref = view.as_view();
        //         self.inner.append_view_unchecked(
        //             view.len(),
        //             first_offset + view_ref.buffer_index(),
        //             view_ref.offset(),
        //         );
        //         // self.inner.append_view(
        //         //     view.len(),
        //         //     first_offset + view_ref.buffer_index(),
        //         //     view_ref.offset(),
        //         // );
        //     }
    }

    // array.buffer(0).into_arrow_buffer()

    //         let buffers_offset = u32::try_from(buffers.len())?;
    //         let canonical_chunk = chunk.clone().into_varbinview()?;
    //         buffers.extend(canonical_chunk.buffers());
    //
    //         for view in canonical_chunk.views().iter() {
    //             if view.is_inlined() {
    //                 // Inlined views can be copied directly into the output
    //                 views.push(*view);
    //             } else {
    //                 // Referencing views must have their buffer_index adjusted with new offsets
    //                 let view_ref = view.as_view();
    //                 views.push(BinaryView::new_view(
    //                     view.len(),
    //                     *view_ref.prefix(),
    //                     buffers_offset + view_ref.buffer_index(),
    //                     view_ref.offset(),
    //                 ));
    //             }
    //         }

    // let array = array.into_canonical()?;
    //         let Canonical::Bool(array) = array else {
    //             vortex_bail!("Expected Canonical::Bool, found {:?}", array);
    //         };
    //
    //         self.inner.append_buffer(&array.boolean_buffer());
    //
    //         match array.validity_mask()?.boolean_buffer() {
    //             AllOr::All => {
    //                 self.append_non_nulls(array.len());
    //                 // If the array is all valid and this builder is non-nullable,
    //                 // we don't need to do anything
    //             }
    //             AllOr::None => {
    //                 self.append_nulls(array.len());
    //             }
    //             AllOr::Some(validity) => {
    //                 if let Some(nulls) = &mut self.nulls {
    //                     nulls.append_buffer(validity.clone())
    //                 } else {
    //                     vortex_bail!("Cannot append nulls to non-nullable builder")
    //                 }
    //             }
    //         }
    //
    //         Ok(())
    // }

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
            buffers
                .iter()
                .map(|b| ByteBuffer::from_arrow_buffer(b.clone(), Alignment::of::<u8>()))
                .collect::<Vec<_>>(),
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
}
