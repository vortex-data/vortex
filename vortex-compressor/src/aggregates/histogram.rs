// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-width counts used to choose BitPacking widths.
//!
//! The histogram counts every row. Null slots and valid zeroes contribute to bin zero. Signed
//! values use their native bit patterns, so negative values contribute to the full-width bin.

use std::sync::Arc;

use itertools::Itertools;
use num_traits::PrimInt;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_session::registry::CachedId;

/// Count the rows requiring each native integer bit width.
///
/// Applies to integer arrays. The partial state contains `ptype.bit_width() + 1` bins, indexed by
/// bit width, and the final result is the same histogram. Partial and final scalars are non-null
/// fixed-size lists of non-null `u64` counts. An empty input produces all-zero bins.
#[derive(Clone, Debug)]
pub struct BitWidthHistogram;

impl AggregateFnVTable for BitWidthHistogram {
    type Options = EmptyOptions;
    type Partial = Vec<usize>;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.compressor.bit_width_histogram");
        *ID
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let DType::Primitive(ptype, _) = input_dtype else {
            return None;
        };
        if !ptype.is_int() {
            return None;
        }

        let bin_count = u32::try_from(ptype.bit_width() + 1).ok()?;

        Some(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, NonNullable)),
            bin_count,
            NonNullable,
        ))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        Ok(vec![0; args.dtype.as_ptype().bit_width() + 1])
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        vortex_ensure!(
            scalar.dtype() == args.partial_dtype,
            "Expected histogram partial dtype {}, got {}",
            args.partial_dtype,
            scalar.dtype()
        );

        scalar
            .as_list()
            .elements()
            .vortex_expect("The histogram partial dtype is a non-null fixed-size list")
            .into_iter()
            .map(|count| {
                let count = count
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("The histogram partial element dtype is non-null u64");

                usize::try_from(count).map_err(|_| {
                    vortex_err!("Expected a histogram count fitting usize, got {count}")
                })
            })
            .collect()
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        for (left, right) in first.iter_mut().zip_eq(second) {
            *left = left
                .checked_add(right)
                .ok_or_else(|| vortex_err!("Histogram count exceeds usize::MAX"))?;
        }

        Ok(first)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let counts = partial
            .iter()
            .map(|&count| {
                let count = u64::try_from(count).map_err(|_| {
                    vortex_err!("Expected a histogram count fitting u64, got {count}")
                })?;

                Ok(Scalar::primitive(count, NonNullable))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let scalar = Scalar::fixed_size_list(
            DType::Primitive(PType::U64, NonNullable),
            counts,
            NonNullable,
        );
        vortex_ensure!(
            scalar.dtype() == args.partial_dtype,
            "Expected histogram partial dtype {}, got {}",
            args.partial_dtype,
            scalar.dtype()
        );

        Ok(scalar)
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Bounding the total row count keeps every bin addition in the kernels representable.
        partial
            .iter()
            .try_fold(batch.len(), |total, &count| total.checked_add(count))
            .ok_or_else(|| vortex_err!("Histogram row count exceeds usize::MAX"))?;

        match batch {
            Columnar::Constant(constant) => {
                let bit_width = if constant.scalar().is_null() {
                    0
                } else {
                    match_each_integer_ptype!(args.dtype.as_ptype(), |T| {
                        let value = constant
                            .scalar()
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("A non-null integer constant has a native value");

                        T::PTYPE.bit_width() - PrimInt::leading_zeros(value) as usize
                    })
                };
                partial[bit_width] += constant.len();
            }
            Columnar::Canonical(canonical) => {
                let array = canonical.as_primitive();
                match_each_integer_ptype!(array.ptype(), |T| {
                    accumulate_primitive::<T>(partial, array, ctx)?;
                });
            }
        }

        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

/// Accumulate a primitive buffer with its validity materialized once.
fn accumulate_primitive<T: NativePType + PrimInt>(
    partial: &mut [usize],
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let bit_width: fn(T) -> usize =
        |value| T::PTYPE.bit_width() - PrimInt::leading_zeros(value) as usize;
    let array = array.as_view();
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;

    match validity.bit_buffer() {
        AllOr::All => {
            for &value in array.as_slice::<T>() {
                partial[bit_width(value)] += 1;
            }
        }
        AllOr::None => {
            partial[0] += array.len();
        }
        AllOr::Some(buffer) => {
            for (is_valid, &value) in buffer.iter().zip_eq(array.as_slice::<T>()) {
                if is_valid {
                    partial[bit_width(value)] += 1;
                } else {
                    partial[0] += 1;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use num_traits::PrimInt;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateDTypes;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::array_session;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::BitWidthHistogram;

    /// Run the aggregate and its partial scalar round trip over one array.
    fn histogram(array: &ArrayRef) -> VortexResult<Vec<usize>> {
        let dtypes =
            AggregateDTypes::try_new(&BitWidthHistogram, &EmptyOptions, array.dtype().clone())?;
        let mut acc = Accumulator::try_new(BitWidthHistogram, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(array, &mut array_session().create_execution_ctx())?;
        let scalar = acc.partial_scalar()?;
        assert_eq!(scalar, acc.final_scalar()?);

        BitWidthHistogram.partial_from_scalar(dtypes.args(&EmptyOptions), scalar)
    }

    /// Find the highest set bit by inspecting every native bit independently.
    fn reference_histogram<T: PrimInt>(values: &[T], validity: &[bool]) -> Vec<usize> {
        let native_width = size_of::<T>() * 8;
        let mut bins = vec![0; native_width + 1];

        for (&value, &is_valid) in values.iter().zip(validity) {
            let mut bit_width = 0;
            if is_valid {
                for bit in 0..native_width {
                    if value.unsigned_shr(u32::try_from(bit).unwrap()) & T::one() != T::zero() {
                        bit_width = bit + 1;
                    }
                }
            }
            bins[bit_width] += 1;
        }

        bins
    }

    #[rstest]
    #[case::u8(PType::U8)]
    #[case::u16(PType::U16)]
    #[case::u32(PType::U32)]
    #[case::u64(PType::U64)]
    #[case::i8(PType::I8)]
    #[case::i16(PType::I16)]
    #[case::i32(PType::I32)]
    #[case::i64(PType::I64)]
    fn matches_native_bit_reference(#[case] ptype: PType) -> VortexResult<()> {
        match_each_integer_ptype!(ptype, |T| {
            let values = [T::MIN, T::MAX, 0, 1, 2, 3];

            for validity in [
                [true; 6],
                [false; 6],
                [true, false, true, false, true, true],
            ] {
                let array = PrimitiveArray::new(
                    Buffer::copy_from(values),
                    validity.into_iter().collect::<Validity>(),
                )
                .into_array();

                assert_eq!(histogram(&array)?, reference_histogram(&values, &validity));
            }
        });

        Ok(())
    }

    #[rstest]
    #[case::null(None, 0)]
    #[case::zero(Some(0), 0)]
    #[case::positive(Some(3), 2)]
    #[case::negative(Some(-1), 16)]
    fn constant_counts(#[case] value: Option<i16>, #[case] bin: usize) -> VortexResult<()> {
        let scalar = value.map_or_else(
            || Scalar::null(DType::Primitive(PType::I16, Nullability::Nullable)),
            |value| Scalar::primitive(value, Nullability::Nullable),
        );
        let array = ConstantArray::new(scalar, 11).into_array();
        let mut expected = vec![0; 17];
        expected[bin] = 11;

        assert_eq!(histogram(&array)?, expected);

        Ok(())
    }

    #[test]
    fn merges_match_whole_array_with_empty_identity() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([
            Some(-1i32),
            Some(0),
            None,
            None,
            Some(2),
            Some(i32::MIN),
            Some(i32::MAX),
        ])
        .into_array();
        let dtypes =
            AggregateDTypes::try_new(&BitWidthHistogram, &EmptyOptions, array.dtype().clone())?;
        let args = dtypes.args(&EmptyOptions);
        let expected = histogram(&array)?;
        let empty = BitWidthHistogram.empty_partial(args)?;

        for start in 0..=array.len() {
            for end in start..=array.len() {
                let first = histogram(&array.slice(0..start)?)?;
                let second = histogram(&array.slice(start..end)?)?;
                let third = histogram(&array.slice(end..array.len())?)?;
                let left = BitWidthHistogram.merge_partials(args, first.clone(), second.clone())?;
                let right = BitWidthHistogram.merge_partials(args, second, third.clone())?;

                assert_eq!(
                    BitWidthHistogram.merge_partials(args, left, third)?,
                    expected
                );
                assert_eq!(
                    BitWidthHistogram.merge_partials(args, first, right)?,
                    expected
                );
            }
        }

        assert_eq!(histogram(&array.slice(0..0)?)?, empty);
        assert_eq!(
            BitWidthHistogram.merge_partials(args, empty.clone(), expected.clone())?,
            expected
        );
        assert_eq!(
            BitWidthHistogram.merge_partials(args, expected.clone(), empty)?,
            expected
        );

        Ok(())
    }
}
