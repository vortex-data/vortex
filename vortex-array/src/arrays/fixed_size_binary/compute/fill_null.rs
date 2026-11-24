// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::buffer::BufferHandle;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::fill_null::FillNullKernel;
use crate::validity::Validity;

impl FillNullKernel for FixedSizeBinary {
    fn fill_null(
        array: ArrayView<'_, FixedSizeBinary>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Validity::Array(is_valid) = array.validity()? else {
            unreachable!("checked in entry point");
        };
        let is_valid = is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer();
        let fill = fill_value
            .as_binary()
            .value()
            .vortex_expect("fill_null requires a non-null fill value");

        let mut values = FixedSizeBinaryArrayExt::values(&array).into_mut();
        fill_invalid_records(
            values.as_mut_slice(),
            array.byte_width() as usize,
            &is_valid,
            fill.as_slice(),
        );

        // SAFETY: Filling replaces whole records, so the buffer keeps its length, and the new
        // validity has no length of its own.
        Ok(Some(
            unsafe {
                FixedSizeBinaryArray::new_unchecked_handle(
                    BufferHandle::new_host(values.freeze()),
                    array.byte_width(),
                    array.len(),
                    Validity::from(fill_value.dtype().nullability()),
                )
            }
            .into_array(),
        ))
    }
}

/// Overwrites every record of `width` bytes whose `is_valid` bit is unset with `fill`.
fn fill_invalid_records(values: &mut [u8], width: usize, is_valid: &BitBuffer, fill: &[u8]) {
    for (index, valid) in is_valid.iter().enumerate() {
        if !valid {
            values[index * width..(index + 1) * width].copy_from_slice(fill);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeBinaryArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn fill_null_replaces_null_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeBinaryArray::from_option_iter([None, Some(b"ab"), None], 2);
        let filled = array.into_array().fill_null(Scalar::fixed_size_binary(
            b"zz".to_vec(),
            Nullability::NonNullable,
        ))?;
        let expected = FixedSizeBinaryArray::from_iter([b"zz", b"ab", b"zz"], 2);
        assert_arrays_eq!(filled, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn fill_null_with_nullable_fill_keeps_nullable_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), None], 2);
        let filled = array.into_array().fill_null(Scalar::fixed_size_binary(
            b"zz".to_vec(),
            Nullability::Nullable,
        ))?;
        let expected = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), Some(b"zz")], 2);
        assert_arrays_eq!(filled, expected, &mut ctx);
        Ok(())
    }
}
