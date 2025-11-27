// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_compute::filter::Filter;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::binaryview::BinaryVector;
use vortex_vector::binaryview::BinaryViewTypeUpcast;
use vortex_vector::binaryview::StringVector;
use vortex_vector::Datum;

use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::execution::kernel;
use crate::execution::BatchKernelRef;
use crate::execution::BindCtx;
use crate::vtable::OperatorVTable;
use crate::vtable::ValidityHelper;
use crate::ArrayRef;

impl OperatorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn bind(
        array: &VarBinViewArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;
        let dtype = array.dtype().clone();

        let views = array.views().clone();
        let buffers: Vec<ByteBuffer> = array.buffers().iter().cloned().collect();
        let buffers = Arc::new(buffers.into_boxed_slice());

        Ok(kernel(move || {
            let selection = mask.execute()?;
            let validity = validity.execute()?;

            // We only filter the views buffer
            let views = views.filter(&selection);

            match dtype {
                // SAFETY: the incoming array has the same validation as the vector
                DType::Utf8(_) => Ok(Datum::from_string(unsafe {
                    StringVector::new_unchecked(views, buffers, validity)
                })),

                // SAFETY: the incoming array has the same validation as the vector
                DType::Binary(_) => Ok(Datum::from_binary(unsafe {
                    BinaryVector::new_unchecked(views, buffers, validity)
                })),
                _ => unreachable!("invalid dtype for VarBinViewArray {dtype}"),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::arrays::BoolArray;
    use crate::arrays::VarBinViewArray;
    use crate::builders::ArrayBuilder;
    use crate::builders::VarBinViewBuilder;
    use crate::IntoArray;

    #[fixture]
    fn strings() -> VarBinViewArray {
        let mut strings = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 5);
        strings.append_value("inlined");
        strings.append_nulls(1);
        strings.append_value("large string 1");
        strings.append_value("large string 2");
        strings.append_value("large string 3");
        strings.finish_into_varbinview()
    }

    #[rstest]
    fn test_bind(strings: VarBinViewArray) {
        // Attempt to bind with a full selection.
        let strings_vec = strings
            .bind(None, &mut ())
            .unwrap()
            .execute()
            .unwrap()
            .into_string();
        assert_eq!(strings_vec.get_ref(0), Some("inlined"));
        assert_eq!(strings_vec.get_ref(1), None);
        assert_eq!(strings_vec.get_ref(2), Some("large string 1"));
        assert_eq!(strings_vec.get_ref(3), Some("large string 2"));
        assert_eq!(strings_vec.get_ref(4), Some("large string 3"));
    }

    #[rstest]
    fn test_bind_with_selection(strings: VarBinViewArray) {
        let selection = BoolArray::from_iter([false, true, false, true, true]).into_array();
        let strings_vec = strings
            .bind(Some(&selection), &mut ())
            .unwrap()
            .execute()
            .unwrap()
            .into_string();

        assert_eq!(strings_vec.get_ref(0), None);
        assert_eq!(strings_vec.get_ref(1), Some("large string 2"));
        assert_eq!(strings_vec.get_ref(2), Some("large string 3"));
    }
}
