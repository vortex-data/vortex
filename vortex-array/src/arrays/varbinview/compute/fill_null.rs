// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;
use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

// Points every null row at one view of the fill value. The data buffers are reused as they are,
// so only the views are copied, plus one buffer for a fill value too long to inline.
impl FillNullKernel for VarBinView {
    fn fill_null(
        array: ArrayView<'_, VarBinView>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let fill_bytes = match fill_value.dtype() {
            DType::Utf8(_) => fill_value
                .as_utf8()
                .value()
                .cloned()
                .map(BufferString::into_inner),
            DType::Binary(_) => fill_value.as_binary().value().cloned(),
            _ => None,
        }
        .ok_or_else(|| vortex_err!("Fill value must be a non-null utf8 or binary scalar"))?;

        let Validity::Array(is_valid) = array.validity()? else {
            unreachable!("checked in entry point");
        };
        let is_invalid = is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer().not();

        let mut buffers: Vec<ByteBuffer> = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host().clone())
            .collect();
        let fill_view =
            BinaryView::make_view(fill_bytes.as_slice(), u32::try_from(buffers.len())?, 0);
        if !fill_view.is_inlined() {
            buffers.push(fill_bytes);
        }

        let mut views = BufferMut::copy_from_in(array.views(), ctx.allocator().clone());
        for (start, end) in is_invalid.set_slices() {
            views[start..end].fill(fill_view);
        }

        let nullability = fill_value.dtype().nullability();
        // SAFETY: the views of valid rows are unchanged. Null rows now hold the fill value's view,
        // which is either inlined or points at the buffer pushed above. The fill value has the
        // array's dtype, so it is valid UTF-8 whenever the array has to be.
        let filled = unsafe {
            VarBinViewArray::new_unchecked(
                views.freeze(),
                Arc::from(buffers),
                array.dtype().with_nullability(nullability),
                Validity::from(nullability),
            )
        };
        Ok(Some(filled.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    const LONG: &str = "a value that is too long to inline";

    #[rstest]
    #[case::inlined_fill("fill", 0)]
    #[case::outlined_fill("a fill value that is too long to inline", 1)]
    fn fill_null_utf8(#[case] fill: &str, #[case] added_buffers: usize) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array =
            VarBinViewArray::from_iter_nullable_str([None, Some("short"), None, Some(LONG), None]);
        let input_buffers = array.data_buffers().len();

        let filled = array
            .into_array()
            .fill_null(Scalar::utf8(fill, Nullability::NonNullable))?
            .execute::<VarBinViewArray>(&mut ctx)?;

        assert_eq!(filled.dtype(), &DType::Utf8(Nullability::NonNullable));
        assert_eq!(filled.data_buffers().len(), input_buffers + added_buffers);
        assert_arrays_eq!(
            filled,
            VarBinViewArray::from_iter_str([fill, "short", fill, LONG, fill]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn fill_null_binary_with_nullable_fill() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinViewArray::from_iter_nullable_bin([Some(b"a".as_slice()), None]);

        let filled = array
            .into_array()
            .fill_null(Scalar::binary(
                LONG.as_bytes().to_vec(),
                Nullability::Nullable,
            ))?
            .execute::<VarBinViewArray>(&mut ctx)?;

        assert_eq!(filled.dtype(), &DType::Binary(Nullability::Nullable));
        assert!(filled.validity()?.definitely_no_nulls());
        assert_arrays_eq!(
            filled,
            VarBinViewArray::from_iter_nullable_bin([Some(b"a".as_slice()), Some(LONG.as_bytes())]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn fill_null_varbin_input() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinArray::from_nullable_strs(vec![Some("a"), None, Some(LONG)]);

        let filled = array
            .into_array()
            .fill_null(Scalar::utf8("fill", Nullability::NonNullable))?
            .execute::<VarBinViewArray>(&mut ctx)?;

        assert_arrays_eq!(
            filled,
            VarBinViewArray::from_iter_str(["a", "fill", LONG]),
            &mut ctx
        );
        Ok(())
    }
}
