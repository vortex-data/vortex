// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::stats::majority::majority_candidate;

impl PrimitiveArray {
    /// Returns the majority candidate of this array's present values and the exact number of
    /// times it occurs.
    ///
    /// The value is found by a Boyer–Moore majority vote, so it is the most common present value
    /// whenever that value fills more than half of the present positions. Otherwise it is some
    /// present value, which callers can judge by its count. Returns `None` if no value is present.
    pub fn majority_candidate(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(PValue, usize)>> {
        if self.is_empty() || self.validity()?.definitely_all_null() {
            return Ok(None);
        }

        let mask = self.as_ref().validity()?.execute_mask(self.len(), ctx)?;
        let top: Option<PValue> = match_each_native_ptype!(self.ptype(), |P| {
            majority_candidate(self.as_slice::<P>(), &mask).map(PValue::from)
        });
        let Some(top) = top else {
            return Ok(None);
        };

        let top_array = ConstantArray::new(
            Scalar::primitive_value(top, top.ptype(), self.dtype().nullability()),
            self.len(),
        );
        let equal = self
            .clone()
            .into_array()
            .compare(top_array.into_array(), CompareOperator::Eq)?
            .execute::<BoolArray>(ctx)?
            .to_bit_buffer();
        let count = match mask.bit_buffer() {
            AllOr::All => equal.true_count(),
            AllOr::None => 0,
            AllOr::Some(validity) => (&equal & validity).true_count(),
        };
        Ok(Some((top, count)))
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation, reason = "test inputs are small")]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::validity::Validity;

    fn candidate(array: &PrimitiveArray) -> VortexResult<Option<(PValue, usize)>> {
        array.majority_candidate(&mut VortexSession::empty().create_execution_ctx())
    }

    /// The majority sits at the end of a sorted array, so a prefix of the input never sees it.
    #[rstest]
    #[case::shorter_than_lanes(5)]
    #[case::not_a_multiple_of_64(1_000)]
    #[case::long(100_003)]
    fn sorted_majority(#[case] len: usize) -> VortexResult<()> {
        let minority = len / 3;
        let values: Buffer<u32> = (0..len)
            .map(|index| {
                if index < minority {
                    index as u32
                } else {
                    u32::MAX
                }
            })
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        assert_eq!(candidate(&array)?, Some((u32::MAX.into(), len - minority)));
        Ok(())
    }

    /// Without a majority, the value is some present value, reported with its exact count.
    #[test]
    fn without_majority_counts_the_returned_value() -> VortexResult<()> {
        let values: Vec<i64> = (0..1_000).map(|index| index % 10).collect();
        let array = PrimitiveArray::new(
            values.iter().copied().collect::<Buffer<i64>>(),
            Validity::NonNullable,
        );
        let (top, count) = candidate(&array)?.vortex_expect("array has valid values");
        let top = i64::try_from(top)?;
        assert_eq!(count, values.iter().filter(|&&value| value == top).count());
        Ok(())
    }

    /// Invalid values must not vote: zero is the most common stored value, but most zeros are null.
    #[rstest]
    #[case::short(104)]
    #[case::long(10_000)]
    fn nulls_do_not_vote(#[case] len: usize) -> VortexResult<()> {
        let values: Buffer<u16> = (0..len)
            .map(|index| if index % 8 < 3 { 9 } else { 0 })
            .collect();
        let validity = Validity::from_iter((0..len).map(|index| index % 8 < 4));
        let array = PrimitiveArray::new(values, validity);
        assert_eq!(candidate(&array)?, Some((9u16.into(), len / 8 * 3)));
        Ok(())
    }

    /// Floats compare by bit pattern, so NaNs are equal and `-0.0` differs from `0.0`.
    #[test]
    fn floats_compare_bits() -> VortexResult<()> {
        let values: Buffer<f64> = [f64::NAN, -0.0, f64::NAN, 0.0, f64::NAN, -0.0, f64::NAN]
            .into_iter()
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let (top, count) = candidate(&array)?.vortex_expect("array has valid values");
        assert!(f64::try_from(top)?.is_nan());
        assert_eq!(count, 4);

        let values: Buffer<f64> = [0.0, -0.0, -0.0].into_iter().collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let (top, count) = candidate(&array)?.vortex_expect("array has valid values");
        assert_eq!(f64::try_from(top)?.to_bits(), (-0.0f64).to_bits());
        assert_eq!(count, 2);
        Ok(())
    }

    #[rstest]
    #[case::all_valid(Validity::NonNullable)]
    #[case::nullable(Validity::from_iter((0..1_000).map(|index| index % 3 != 0)))]
    fn bytes(#[case] validity: Validity) -> VortexResult<()> {
        let values: Buffer<i8> = (0..1_000)
            .map(|index| {
                if index % 3 == 0 {
                    (index % 7) as i8
                } else {
                    -5
                }
            })
            .collect();
        // Every `-5` is valid in both cases.
        let array = PrimitiveArray::new(values, validity);
        assert_eq!(candidate(&array)?, Some(((-5i8).into(), 666)));
        Ok(())
    }

    #[test]
    fn all_null() -> VortexResult<()> {
        let array = PrimitiveArray::new(Buffer::<u32>::zeroed(10), Validity::AllInvalid);
        assert_eq!(candidate(&array)?, None);
        Ok(())
    }
}
