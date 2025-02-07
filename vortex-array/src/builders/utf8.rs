use std::any::Any;

use arrow_array::builder::{ArrayBuilder as _, StringViewBuilder};
use arrow_array::{Array as ArrowArray, StringViewArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::array::VarBinViewArray;
use crate::arrow::FromArrowArray;
use crate::builders::ArrayBuilder;
use crate::{Array, Canonical, IntoCanonical};

pub struct Utf8Builder {
    inner: StringViewBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl Utf8Builder {
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: StringViewBuilder::with_capacity(capacity),
            nullability,
            dtype: DType::Utf8(nullability),
        }
    }

    pub fn append_value<S: AsRef<str>>(&mut self, value: S) {
        self.inner.append_value(value.as_ref())
    }

    pub fn append_option<S: AsRef<str>>(&mut self, value: Option<S>) {
        self.inner.append_option(value.as_ref())
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
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.inner.append_value("")
        }
    }

    fn append_nulls(&mut self, n: usize) {
        for _ in 0..n {
            self.inner.append_null()
        }
    }

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = if let Some(array) = VarBinViewArray::maybe_from(&array) {
            array
        } else {
            let Canonical::VarBinView(array) = array.into_canonical() else {
                vortex_bail!("Expected Canonical::VarBinView, found {:?}", array);
            };
            array
        };

        let mut first_offset = None;
        for buf in array.buffers() {
            let offset = self.inner.append_block(buf.into_arrow_buffer());
            if first_offset.is_none() {
                first_offset = Some(offset);
            }
        }
        let first_offset = first_offset.unwrap_or(0);

        for view in array.views().iter() {
            if view.is_inlined() {
                self.inner.append_value(*view);
            } else {
                let view_ref = view.as_view();
                self.inner.append_view_unchecked(
                    view.len(),
                    first_offset + view_ref.buffer_index(),
                    view_ref.offset(),
                );
                // self.inner.append_view(
                //     view.len(),
                //     first_offset + view_ref.buffer_index(),
                //     view_ref.offset(),
                // );
            }
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
        todo!("array {}", array.tree_display())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        let arrow = self.inner.finish();

        if !self.dtype().is_nullable() && arrow.null_count() > 0 {
            vortex_bail!("Non-nullable builder has null values");
        }

        Ok(Array::from_arrow(&arrow, self.nullability.into()))
    }
}
