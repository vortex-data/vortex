// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BitBufferView;
use vortex_buffer::get_bit;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::builtins::ArrayBuiltins;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;

impl TakeExecute for Bool {
    fn take(
        array: ArrayView<'_, Bool>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let indices_nulls_zeroed = match indices.validity()?.execute_mask(indices.len(), ctx)? {
            Mask::AllTrue(_) => indices.clone(),
            Mask::AllFalse(_) => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                        .into_array(),
                ));
            }
            Mask::Values(_) => indices
                .clone()
                .fill_null(Scalar::from(0).cast(indices.dtype())?)?,
        };
        let indices_nulls_zeroed = indices_nulls_zeroed.execute::<PrimitiveArray>(ctx)?;
        let buffer = match_each_integer_ptype!(indices_nulls_zeroed.ptype(), |I| {
            take_valid_indices(
                array.bit_buffer_view(),
                indices_nulls_zeroed.as_slice::<I>(),
            )
        });

        Ok(Some(
            BoolArray::new(buffer, array.validity()?.take(indices)?).into_array(),
        ))
    }
}

fn take_contiguous_ranges(
    array: ArrayView<'_, Bool>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };
    let source = array.to_bit_buffer();
    let output_len = indices_ref.len();
    let buffer = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                take_bit_slices_constant_length(
                    &source,
                    starts.as_slice::<S>(),
                    length,
                    output_len,
                )?
            })
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                    take_bit_slices(
                        &source,
                        starts.as_slice::<S>(),
                        lengths.as_slice::<L>(),
                        output_len,
                    )?
                })
            })
        }
    };

    Ok(Some(
        BoolArray::new(buffer, array.validity()?.take(indices_ref)?).into_array(),
    ))
}

fn take_valid_indices<I: AsPrimitive<usize>>(bools: BitBufferView<'_>, indices: &[I]) -> BitBuffer {
    // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
    // the overhead to convert to a Vec<bool>.
    if bools.len() <= 4096 {
        let bools = bools.iter().collect_vec();
        take_byte_bool(bools, indices)
    } else {
        take_bool_impl(bools, indices)
    }
}

fn take_byte_bool<I: AsPrimitive<usize>>(bools: Vec<bool>, indices: &[I]) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        bools[unsafe { indices.get_unchecked(idx).as_() }]
    })
}

fn take_bool_impl<I: AsPrimitive<usize>>(bools: BitBufferView<'_>, indices: &[I]) -> BitBuffer {
    // We dereference to underlying buffer to avoid access cost on every index.
    let buffer = bools.inner();
    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: we can take from the indices unchecked since collect_bool just iterates len.
        let idx = unsafe { indices.get_unchecked(idx).as_() };
        get_bit(buffer, bools.offset() + idx)
    })
}

fn take_bit_slices_constant_length<S>(
    source: &BitBuffer,
    starts: &[S],
    length: usize,
    output_len: usize,
) -> VortexResult<BitBuffer>
where
    S: UnsignedPType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );

    let mut values = BitBufferMut::with_capacity(output_len);
    for start in starts {
        let start = start.as_();
        values.append_buffer(&source.slice(start..).slice(..length));
    }

    Ok(values.freeze())
}

fn take_bit_slices<S, L>(
    source: &BitBuffer,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<BitBuffer>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    let mut values = BitBufferMut::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        values.append_buffer(&source.slice(start..).slice(..length));
    }

    vortex_ensure!(
        values.len() == output_len,
        "PiecewiseSequenceArray expanded length {} does not match declared length {output_len}",
        values.len()
    );
    Ok(values.freeze())
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, reason = "test-sized index values")]
mod test {
    use itertools::Itertools as _;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PiecewiseSequenceArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    /// Contiguous runs with per-piece lengths, taking the `Columnar::Canonical` branch of
    /// `take_contiguous_ranges`.
    fn contiguous_indices(starts: &[u64], lengths: &[u64]) -> VortexResult<ArrayRef> {
        let len = lengths.iter().sum::<u64>() as usize;
        Ok(PiecewiseSequenceArray::try_new(
            PrimitiveArray::from_iter(starts.iter().copied()).into_array(),
            PrimitiveArray::from_iter(lengths.iter().copied()).into_array(),
            ConstantArray::new(1u64, starts.len()).into_array(),
            len,
        )?
        .into_array())
    }

    /// Contiguous runs of equal length, taking the `Columnar::Constant` branch of
    /// `take_contiguous_ranges`.
    fn contiguous_indices_constant_length(starts: &[u64], length: u64) -> VortexResult<ArrayRef> {
        let len = starts.len() * length as usize;
        Ok(PiecewiseSequenceArray::try_new(
            PrimitiveArray::from_iter(starts.iter().copied()).into_array(),
            ConstantArray::new(length, starts.len()).into_array(),
            ConstantArray::new(1u64, starts.len()).into_array(),
            len,
        )?
        .into_array())
    }

    fn alternating_bools(len: usize) -> Vec<bool> {
        (0..len).map(|idx| idx % 3 == 0).collect()
    }

    #[test]
    fn take_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = reference
            .take(buffer![0, 3, 4].into_array())
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            b.to_bit_buffer(),
            BoolArray::from_iter([Some(false), None, Some(false)]).to_bit_buffer()
        );

        let all_invalid_indices = PrimitiveArray::from_option_iter([None::<i32>, None, None]);
        let b = reference.take(all_invalid_indices.into_array()).unwrap();
        assert_arrays_eq!(b, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    #[test]
    fn test_bool_array_take_with_null_out_of_bounds_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();

        // position 3 is null, the third index is null
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), None, None]),
            &mut ctx
        );
    }

    #[test]
    fn test_non_null_bool_array_take_with_null_out_of_bounds_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        // the third index is null
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), Some(true), None]),
            &mut ctx
        );
    }

    #[test]
    fn test_bool_array_take_all_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    #[test]
    fn test_non_null_bool_array_take_all_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    fn expand(starts: &[u64], lengths: &[u64]) -> ArrayRef {
        PrimitiveArray::from_iter(
            starts
                .iter()
                .zip_eq(lengths)
                .flat_map(|(&start, &length)| start..start + length),
        )
        .into_array()
    }

    /// Byte-aligned starts and lengths hit the `copy_from_slice` branch of `append_buffer`.
    #[test]
    fn contiguous_byte_aligned_runs() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(alternating_bools(24)).into_array();

        let taken = values.take(contiguous_indices_constant_length(&[0, 8, 16], 8)?)?;
        assert_arrays_eq!(taken, values, &mut ctx);

        let taken = values.take(contiguous_indices_constant_length(&[16, 0], 8)?)?;
        assert_arrays_eq!(taken, values.take(expand(&[16, 0], &[8, 8]))?, &mut ctx);
        Ok(())
    }

    /// Unaligned starts and lengths fall through to the bit-level copy, which must agree with
    /// the general per-index gather.
    #[rstest]
    #[case(&[3], &[5])]
    #[case(&[0], &[1])]
    #[case(&[3, 11], &[5, 7])]
    #[case(&[9, 1, 20], &[3, 8, 4])]
    #[case(&[7, 7], &[0, 9])]
    fn contiguous_unaligned_runs(
        #[case] starts: &[u64],
        #[case] lengths: &[u64],
        #[values(false, true)] nullable: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bools = alternating_bools(24);
        let values = if nullable {
            BoolArray::from_iter(
                bools
                    .iter()
                    .enumerate()
                    .map(|(idx, &b)| (idx % 5 != 2).then_some(b)),
            )
        } else {
            BoolArray::from_iter(bools)
        }
        .into_array();

        let expected = values.take(expand(starts, lengths))?;
        assert_arrays_eq!(
            values.take(contiguous_indices(starts, lengths)?)?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    /// A sliced source carries a non-zero bit offset into `take_bit_slices`.
    #[rstest]
    fn contiguous_runs_from_sliced_source(
        #[values(0, 1, 3, 8, 11)] offset: usize,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(alternating_bools(40))
            .into_array()
            .slice(offset..offset + 20)?;

        let starts = [1u64, 9];
        let lengths = [6u64, 5];
        assert_arrays_eq!(
            values.take(contiguous_indices(&starts, &lengths)?)?,
            values.take(expand(&starts, &lengths))?,
            &mut ctx
        );
        Ok(())
    }

    /// Nulls in the source must be gathered alongside the values on the contiguous-range path.
    #[test]
    fn contiguous_runs_preserve_validity() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter([
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            None,
            Some(true),
            Some(false),
        ])
        .into_array();

        assert_arrays_eq!(
            values.take(contiguous_indices(&[1, 6], &[3, 2])?)?,
            BoolArray::from_iter([None, Some(false), Some(true), Some(true), Some(false)]),
            &mut ctx
        );

        // Constant-length pieces take a different branch, so cover validity there too.
        assert_arrays_eq!(
            values.take(contiguous_indices_constant_length(&[4, 0], 2)?)?,
            BoolArray::from_iter([None, None, Some(true), None]),
            &mut ctx
        );
        Ok(())
    }

    /// An all-null source array is still all-null after a contiguous-range take.
    #[test]
    fn contiguous_runs_all_null_source() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::new(BitBuffer::new_unset(8), Validity::AllInvalid).into_array();

        assert_arrays_eq!(
            values.take(contiguous_indices(&[2], &[3])?)?,
            BoolArray::from_iter([None::<bool>, None, None]),
            &mut ctx
        );
        Ok(())
    }

    /// `take_valid_indices` switches from the `Vec<bool>` gather to the bitmap gather at 4096
    /// elements; both must produce the same result, with and without nulls.
    #[rstest]
    fn take_matches_across_page_threshold(
        #[values(64, 4095, 4096, 4097, 9000)] len: usize,
        #[values(false, true)] nullable: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bools = alternating_bools(len);
        let values = if nullable {
            BoolArray::from_iter(
                bools
                    .iter()
                    .enumerate()
                    .map(|(idx, &b)| (idx % 7 != 3).then_some(b)),
            )
        } else {
            BoolArray::from_iter(bools.clone())
        };

        let indices = (0..len).map(|idx| ((idx * 31 + 7) % len) as u64);
        let taken = values
            .into_array()
            .take(PrimitiveArray::from_iter(indices.clone()).into_array())?
            .execute::<BoolArray>(&mut ctx)?;

        let expected = if nullable {
            BoolArray::from_iter(indices.clone().map(|idx| {
                let idx = idx as usize;
                (idx % 7 != 3).then_some(bools[idx])
            }))
        } else {
            BoolArray::from_iter(indices.clone().map(|idx| bools[idx as usize]))
        };
        assert_arrays_eq!(taken.into_array(), expected, &mut ctx);
        Ok(())
    }

    /// Null indices must produce nulls even when the source is gathered via the bitmap path.
    #[test]
    fn nullable_indices_over_large_source() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bools = alternating_bools(5000);
        let values = BoolArray::from_iter(bools.clone()).into_array();
        let indices = PrimitiveArray::new(
            buffer![0u64, 4999, 12345, 17],
            Validity::Array(BoolArray::from_iter([true, true, false, true]).into_array()),
        );

        assert_arrays_eq!(
            values.take(indices.into_array())?,
            BoolArray::from_iter([Some(bools[0]), Some(bools[4999]), None, Some(bools[17]),]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true, false]))]
    #[case(BoolArray::from_iter([true]))]
    fn test_take_bool_conformance(#[case] array: BoolArray) {
        test_take_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
