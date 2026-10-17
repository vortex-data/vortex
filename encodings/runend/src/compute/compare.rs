// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::collect_bool_word;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;
use crate::decompress_bool::runend_decode_bools;
use crate::iter::trimmed_ends_iter;

impl CompareKernel for RunEnd {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            let values = lhs.values().binary(
                ConstantArray::new(const_scalar, lhs.values().len()).into_array(),
                Operator::from(operator),
            )?;
            return runend_decode_bools(
                lhs.ends().clone().execute::<PrimitiveArray>(ctx)?,
                values.execute::<BoolArray>(ctx)?,
                lhs.offset(),
                lhs.len(),
                ctx,
            )
            .map(Some);
        }

        compare_runs(lhs, rhs, operator, ctx)
    }
}

/// Compare each run's value against the other side's values within that run, without decoding
/// the runs.
fn compare_runs(
    lhs: ArrayView<'_, RunEnd>,
    rhs: &ArrayRef,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let DType::Primitive(ptype, _) = lhs.dtype() else {
        return Ok(None);
    };
    let values = lhs.values().clone().execute::<PrimitiveArray>(ctx)?;
    // TODO: expand null runs into the result's validity.
    if !matches!(
        values.validity()?.execute_mask(values.len(), ctx)?,
        Mask::AllTrue(_)
    ) {
        return Ok(None);
    }
    let ends = lhs.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let other = rhs.clone().execute::<PrimitiveArray>(ctx)?;

    let bits = match_each_native_ptype!(*ptype, |T| {
        match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            compare_runs_op(
                trimmed_ends_iter(ends.as_slice::<E>(), lhs.offset(), lhs.len()),
                values.as_slice::<T>(),
                other.as_slice::<T>(),
                operator,
            )
        })
    });

    let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
    let validity = other.validity()?.union_nullability(nullability);
    Ok(Some(BoolArray::new(bits, validity).into_array()))
}

/// Dispatch the operator outside the word loop so each instantiation vectorizes one predicate.
fn compare_runs_op<T: NativePType>(
    ends: impl Iterator<Item = usize>,
    values: &[T],
    other: &[T],
    operator: CompareOperator,
) -> BitBuffer {
    match operator {
        CompareOperator::Eq => compare_runs_typed(ends, values, other, |a: T, b| a.is_eq(b)),
        CompareOperator::NotEq => compare_runs_typed(ends, values, other, |a: T, b| !a.is_eq(b)),
        CompareOperator::Gt => compare_runs_typed(ends, values, other, T::is_gt),
        CompareOperator::Gte => compare_runs_typed(ends, values, other, T::is_ge),
        CompareOperator::Lt => compare_runs_typed(ends, values, other, T::is_lt),
        CompareOperator::Lte => compare_runs_typed(ends, values, other, T::is_le),
    }
}

/// Pack `f(run_value, other[i])` for every position, 64 positions per word.
///
/// A word inside one run compares against that run's value directly. A word that crosses runs
/// first expands its runs into a 64-value scratch buffer.
fn compare_runs_typed<T: NativePType>(
    ends: impl Iterator<Item = usize>,
    values: &[T],
    other: &[T],
    f: impl Fn(T, T) -> bool,
) -> BitBuffer {
    let len = other.len();
    let mut words = BufferMut::<u64>::zeroed(len.div_ceil(64));
    let mut runs = Runs {
        runs: ends.zip(values.iter().copied()),
        end: 0,
        value: T::default(),
    };
    if !runs.advance() {
        return BitBuffer::new_unset(len);
    }
    let mut scratch = [T::default(); 64];

    let (chunks, tail) = other.as_chunks::<64>();
    for (word_idx, (word, chunk)) in words.iter_mut().zip(chunks).enumerate() {
        let start = word_idx * 64;
        // Fixed-size chunks let the compiler prove the predicate's indexing stays in bounds.
        *word = if runs.end >= start + 64 {
            let value = runs.value;
            collect_bool_word(64, |i| f(value, chunk[i]))
        } else {
            runs.expand(&mut scratch, start);
            collect_bool_word(64, |i| f(scratch[i], chunk[i]))
        };
        runs.skip_ended(start + 64);
    }
    if !tail.is_empty() {
        let start = chunks.len() * 64;
        runs.expand(&mut scratch[..tail.len()], start);
        words[chunks.len()] = collect_bool_word(tail.len(), |i| f(scratch[i], tail[i]));
    }

    let mut bytes = words.into_byte_buffer();
    bytes.truncate(len.div_ceil(8));
    BitBuffer::new(bytes.freeze(), len)
}

/// A cursor over `(end, value)` runs, positioned on the run containing the next position.
struct Runs<I, T> {
    runs: I,
    end: usize,
    value: T,
}

impl<I: Iterator<Item = (usize, T)>, T: Copy> Runs<I, T> {
    fn advance(&mut self) -> bool {
        match self.runs.next() {
            Some((end, value)) => {
                (self.end, self.value) = (end, value);
                true
            }
            None => false,
        }
    }

    /// Skip the runs ending at or before `position`.
    fn skip_ended(&mut self, position: usize) {
        while self.end <= position && self.advance() {}
    }

    /// Fill `scratch` with the run values of the positions starting at `start`.
    fn expand(&mut self, scratch: &mut [T], start: usize) {
        let mut pos = 0;
        while pos < scratch.len() {
            let segment_end = (self.end - start).min(scratch.len());
            scratch[pos..segment_end].fill(self.value);
            pos = segment_end;
            if pos < scratch.len() {
                self.advance();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEnd;
    use crate::RunEndArray;
    use crate::array::RunEndArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn ree_array(ctx: &mut ExecutionCtx) -> RunEndArray {
        RunEnd::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
            ctx,
        )
        .unwrap()
    }

    #[test]
    fn compare_run_end() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = ree_array(&mut ctx);
        let res = arr
            .into_array()
            .binary(ConstantArray::new(5, 12).into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([
            false, false, false, false, false, false, false, false, true, true, true, true,
        ]);
        assert_arrays_eq!(res, expected, &mut ctx);
    }

    /// Compares a run-end array against a non-constant array without decoding the runs, for runs
    /// shorter than, aligned with and longer than a 64-value word, sliced and unsliced.
    #[rstest]
    fn compare_runs_against_array(
        #[values(1, 3, 64, 100, 1024)] run_len: u32,
        #[values((0, 5000), (37, 4100))] range: (usize, usize),
        #[values(
            CompareOperator::Eq,
            CompareOperator::NotEq,
            CompareOperator::Lt,
            CompareOperator::Gte
        )]
        operator: CompareOperator,
        #[values(false, true)] nullable_other: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<u32> = (0..5000u32).map(|i| (i / run_len) % 7).collect();
        let encoded = RunEnd::encode(
            PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array(),
            &mut ctx,
        )?;
        // Adjacent runs never repeat a value, so every run is exactly `run_len` long.
        let run_len = run_len as usize;
        let runs = range.0 / run_len..range.1.div_ceil(run_len);
        let runs = RunEnd::try_new_offset_length(
            encoded.ends().slice(runs.clone())?,
            encoded.values().slice(runs)?,
            range.0,
            range.1 - range.0,
            &mut ctx,
        )?;
        let other: Buffer<u32> = (0..5000u32).map(|i| (i * 7919) % 8).collect();
        let validity = if nullable_other {
            Validity::from_iter((0..5000).map(|i| i % 11 != 0))
        } else {
            Validity::NonNullable
        };
        let other = PrimitiveArray::new(other, validity)
            .into_array()
            .slice(range.0..range.1)?;

        let expected = PrimitiveArray::new(values, Validity::NonNullable)
            .into_array()
            .slice(range.0..range.1)?
            .binary(other.clone(), Operator::from(operator))?;
        let result =
            <RunEnd as CompareKernel>::compare(runs.as_view(), &other, operator, &mut ctx)?
                .expect("kernel handles primitive arrays");
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
