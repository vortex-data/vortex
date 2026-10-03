// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::Delta as FastLanesDelta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::delta_decompress::decode_chunk;

const WORDS_PER_CHUNK: usize = 1024 / u64::BITS as usize;

/// Compares against a constant one chunk at a time, so the decoded values stay in a stack buffer
/// and only the result bits are written out.
impl CompareKernel for Delta {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant) = constant.as_primitive_opt() else {
            return Ok(None);
        };
        let ptype = lhs.dtype().as_ptype();
        if constant.ptype() != ptype {
            return Ok(None);
        }

        let bits = match_each_integer_ptype!(ptype, |T| {
            let rhs: T = constant
                .typed_value::<T>()
                .vortex_expect("compare adaptor strips null constants");
            match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
                const LANES: usize = U::LANES;
                compare_constant::<U, LANES>(lhs, rhs.as_(), ptype.is_signed_int(), operator, ctx)?
            })
        });

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = lhs.validity()?.union_nullability(nullability);
        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

/// Compare every value against `rhs` in the unsigned domain the chunks decode into.
///
/// Flipping the sign bit maps signed order onto unsigned order, so ordered comparisons of signed
/// values work on their unsigned bit patterns.
fn compare_constant<U, const LANES: usize>(
    array: ArrayView<'_, Delta>,
    rhs: U,
    signed: bool,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer>
where
    U: NativePType + PrimInt + FastLanesDelta + Transpose,
{
    let bases = array
        .bases()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let deltas = array
        .deltas()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let (bases, deltas) = (bases.as_slice::<U>(), deltas.as_slice::<U>());

    let sign = if signed {
        U::one() << (U::PTYPE.bit_width() - 1)
    } else {
        U::zero()
    };
    let rhs = rhs ^ sign;

    let offset = array.offset();
    let len = array.len();
    let num_chunks = (offset + len).div_ceil(1024);
    let mut words = BufferMut::<u64>::zeroed(num_chunks * WORDS_PER_CHUNK);
    let mut transposed = [U::zero(); 1024];
    let mut values = [U::zero(); 1024];
    for (chunk, out) in words
        .as_mut_slice()
        .as_chunks_mut::<WORDS_PER_CHUNK>()
        .0
        .iter_mut()
        .enumerate()
    {
        decode_chunk::<U, LANES>(bases, deltas, chunk, &mut transposed, &mut values);
        match operator {
            CompareOperator::Eq => pack(out, &values, |v| (v ^ sign) == rhs),
            CompareOperator::NotEq => pack(out, &values, |v| (v ^ sign) != rhs),
            CompareOperator::Lt => pack(out, &values, |v| (v ^ sign) < rhs),
            CompareOperator::Lte => pack(out, &values, |v| (v ^ sign) <= rhs),
            CompareOperator::Gt => pack(out, &values, |v| (v ^ sign) > rhs),
            CompareOperator::Gte => pack(out, &values, |v| (v ^ sign) >= rhs),
        }
    }

    Ok(BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len).freeze())
}

/// Write one bit per value, least significant bit first, 64 values per word.
#[inline]
fn pack<U: Copy>(
    out: &mut [u64; WORDS_PER_CHUNK],
    values: &[U; 1024],
    predicate: impl Fn(U) -> bool,
) {
    for (word, values) in out.iter_mut().zip(values.as_chunks::<64>().0) {
        *word = values.iter().enumerate().fold(0u64, |word, (bit, &value)| {
            word | (u64::from(predicate(value)) << bit)
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Series-major timestamps: two series that each restart, length not a multiple of 1024.
    fn timestamps() -> PrimitiveArray {
        PrimitiveArray::from_iter((0..2).flat_map(|_| (0..1500i64).map(|i| i * 30 - 20_000)))
    }

    #[rstest]
    #[case(Operator::Eq)]
    #[case(Operator::NotEq)]
    #[case(Operator::Lt)]
    #[case(Operator::Lte)]
    #[case(Operator::Gt)]
    #[case(Operator::Gte)]
    fn compare_matches_decoded(#[case] op: Operator) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = timestamps();
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?.into_array();
        // A threshold that is negative, so the signed path matters.
        let rhs = ConstantArray::new(-5_000i64, primitive.len()).into_array();

        let actual = delta
            .binary(rhs.clone(), op)?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = primitive
            .into_array()
            .binary(rhs, op)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn compare_on_slice_and_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive =
            PrimitiveArray::from_option_iter((0u32..3000).map(|v| (v % 7 != 0).then_some(v * 3)));
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?
            .into_array()
            .slice(1000..2500)?;
        let rhs = ConstantArray::new(5_000u32, delta.len()).into_array();

        let actual = delta
            .binary(rhs.clone(), Operator::Gte)?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = primitive
            .into_array()
            .slice(1000..2500)?
            .binary(rhs, Operator::Gte)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
