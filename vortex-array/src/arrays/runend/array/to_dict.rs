// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Express run-end decoding as a gather from the existing values child.
//!
//! Repeated run indices let the generic take machinery handle logical types without specialized
//! run-end decoders. Piecewise codes keep that mapping compact until a consumer needs flat indices.

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::runend::RunEndArray;
use crate::arrays::runend::RunEndArrayExt;
use crate::arrays::runend::RunEndArraySlotsExt;
use crate::arrays::runend::slice_relative_ends;
use crate::dtype::Nullability;
use crate::dtype::UnsignedPType;
use crate::match_each_unsigned_integer_ptype;

impl RunEndArray {
    /// Converts to a dictionary with the same logical dtype and slice.
    ///
    /// Non-empty arrays share the values child without executing it. Codes use one constant piece
    /// per overlapping run, so construction needs no expanded index buffer. Executing the resulting
    /// [`DictArray`] may still materialize those indices. Empty arrays retain no source children.
    pub fn to_dict(&self, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
        if self.is_empty() {
            return DictArray::try_new(
                PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array(),
                Canonical::empty(self.dtype()).into_array(),
            );
        }

        let ends = self.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let codes = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            run_indices(ends.as_slice::<E>(), self.offset(), self.len())?
        });

        DictArray::try_new(codes.into_array(), self.values().clone())
    }
}

/// Builds repeated physical value indices for a slice covered by validated run ends.
fn run_indices<E: UnsignedPType>(
    ends: &[E],
    offset: usize,
    len: usize,
) -> VortexResult<PiecewiseSequenceArray> {
    let mut starts = BufferMut::<u64>::with_capacity(ends.len());
    let mut lengths = BufferMut::<u64>::with_capacity(ends.len());
    let mut run_start = 0;

    for (index, run_end) in slice_relative_ends(ends, offset, len).enumerate() {
        let length = run_end - run_start;
        if length != 0 {
            starts.push(u64::try_from(index)?);
            lengths.push(u64::try_from(length)?);
        }
        run_start = run_end;

        if run_end == len {
            break;
        }
    }

    let multipliers = ConstantArray::new(0u64, starts.len()).into_array();

    PiecewiseSequenceArray::try_new(
        starts.freeze().into_array(),
        lengths.freeze().into_array(),
        multipliers,
        len,
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PiecewiseSequence;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::arrays::runend::RunEndArraySlotsExt;
    use crate::arrays::runend::tests::SESSION;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;

    #[track_caller]
    fn check_conversion(
        array: &RunEndArray,
        expected: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dict = array.to_dict(ctx)?;

        if array.is_empty() {
            assert!(dict.codes().is_empty());
            assert!(dict.values().is_empty());
        } else {
            assert!(ArrayRef::ptr_eq(dict.values(), array.values()));
            let codes = dict.codes().as_::<PiecewiseSequence>();
            assert_eq!(codes.multipliers().as_constant(), Some(&Scalar::from(0u64)));
        }

        assert_arrays_eq!(dict.into_array(), expected, ctx);

        Ok(())
    }

    fn struct_values(storage: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(StructArray::from_fields(&[("value", storage)])?.into_array())
    }

    fn fixed_size_list_values(storage: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(
            FixedSizeListArray::try_new(storage.clone(), 1, storage.validity()?, storage.len())?
                .into_array(),
        )
    }

    fn timestamp_values(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let ext_dtype = Timestamp::try_new_with_tz(
            TimeUnit::Microseconds,
            Some("UTC".into()),
            storage.dtype().nullability(),
        )?;

        Ok(ExtensionArray::try_new(ext_dtype.erased(), storage)?.into_array())
    }

    #[rstest]
    #[case::full(PType::U32, 0, 8)]
    #[case::clipped_u8(PType::U8, 2, 5)]
    #[case::clipped_u16(PType::U16, 2, 5)]
    #[case::clipped_u32(PType::U32, 2, 5)]
    #[case::clipped_u64(PType::U64, 2, 5)]
    #[case::run_boundary(PType::U32, 3, 2)]
    #[case::single_row(PType::U32, 1, 1)]
    #[case::empty_retained_runs(PType::U32, 8, 0)]
    fn sliced_conversion(
        #[case] ends_type: PType,
        #[case] offset: usize,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = buffer![3u64, 5, 8]
            .into_array()
            .cast(DType::Primitive(ends_type, Nullability::NonNullable))?;
        let values = PrimitiveArray::from_option_iter([Some(10i64), None, Some(30)]).into_array();
        let array = RunEnd::try_new_offset_length(ends, values, offset, len, &mut ctx)?;
        let expanded = [
            Some(10i64),
            Some(10),
            Some(10), // First run.
            None,
            None, // Null run.
            Some(30),
            Some(30),
            Some(30), // Last run.
        ];
        let expected =
            PrimitiveArray::from_option_iter(expanded[offset..offset + len].iter().copied());

        check_conversion(&array, expected.into_array(), &mut ctx)
    }

    #[test]
    fn codes_retain_only_overlapping_runs() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 5, 8].into_array(),
            buffer![10i64, 20, 30].into_array(),
            3,
            4,
            &mut ctx,
        )?;
        let dict = array.to_dict(&mut ctx)?;
        let codes = dict.codes().as_::<PiecewiseSequence>();

        assert_arrays_eq!(codes.starts(), buffer![1u64, 2].into_array(), &mut ctx);
        assert_arrays_eq!(codes.lengths(), buffer![2u64, 2].into_array(), &mut ctx);

        assert_arrays_eq!(
            dict.into_array(),
            buffer![20i64, 20, 30, 30].into_array(),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn empty_conversion() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::empty::<i64>(Nullability::Nullable).into_array();
        let array = RunEnd::try_new(
            PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array(),
            values.clone(),
            &mut ctx,
        )?;

        check_conversion(&array, values, &mut ctx)
    }

    #[test]
    fn repeated_slicing() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new(
            buffer![3u32, 5, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(10i64), None, Some(30)]).into_array(),
            &mut ctx,
        )?;
        let sliced = array.slice(2..8)?.execute_until::<RunEnd>(&mut ctx)?;
        let sliced = sliced
            .slice(1..5)?
            .execute_until::<RunEnd>(&mut ctx)?
            .downcast::<RunEnd>();
        let expected = PrimitiveArray::from_option_iter([None, None, Some(30i64), Some(30)]);

        check_conversion(&sliced, expected.into_array(), &mut ctx)
    }

    #[rstest]
    #[case::structure(struct_values)]
    #[case::fixed_size_list(fixed_size_list_values)]
    #[case::extension(timestamp_values)]
    fn fallback_preserves_dtype(
        #[case] wrap: fn(ArrayRef) -> VortexResult<ArrayRef>,
        #[values(false, true)] empty: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let storage = PrimitiveArray::from_option_iter([Some(10i64), None, Some(30)]).into_array();
        let values = wrap(storage)?;
        let len = if empty { 0 } else { 5 };
        let expected = wrap(
            PrimitiveArray::from_option_iter(
                [Some(10i64), None, None, Some(30), Some(30)]
                    .into_iter()
                    .take(len),
            )
            .into_array(),
        )?;
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 5, 8].into_array(),
            values,
            2,
            len,
            &mut ctx,
        )?;

        check_conversion(&array, expected.clone(), &mut ctx)?;
        assert_arrays_eq!(array.into_array(), expected, &mut ctx);

        Ok(())
    }
}
