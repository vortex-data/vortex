// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::repeat_n;

use arrow_array::cast::AsArray;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::read_u64_le;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrow::ArrowSessionExt;
use crate::arrow::FromArrowArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::scalar::BoolScalar;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// Trait for encoding-specific boolean kernels that operate in encoded space.
///
/// Implementations receive the encoded array as the left operand. `rhs` may be any boolean array
/// encoding or a constant; implementations should return `Ok(None)` when they cannot handle that
/// operand without falling back to ordinary execution.
///
/// Vortex's boolean [`Operator::And`] and [`Operator::Or`] variants use Kleene semantics; there is
/// no separate two-valued boolean operator path to dispatch here.
pub trait BooleanKernel: VTable {
    /// Execute `lhs <operator> rhs` using Kleene boolean semantics.
    fn boolean(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: Operator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that bridges [`BooleanKernel`] implementations to [`ExecuteParentKernel`].
///
/// When a `ScalarFnArray(Binary, And|Or)` wraps a child implementing [`BooleanKernel`], this
/// adaptor extracts the other operand and delegates to the encoding-specific kernel.
#[derive(Default, Debug)]
pub struct BooleanExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for BooleanExecuteAdaptor<V>
where
    V: BooleanKernel,
{
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let op = *parent.options;
        if !is_boolean_operator(op) {
            return Ok(None);
        }

        let Some(scalar_fn_array) = parent.as_opt::<ScalarFn>() else {
            return Ok(None);
        };
        let other = match child_idx {
            0 => scalar_fn_array.get_child(1),
            1 => scalar_fn_array.get_child(0),
            _ => return Ok(None),
        };

        if let Some(result) = constant_boolean(array.array(), other, op)? {
            return Ok(Some(result));
        }

        V::boolean(array, other, op, ctx)
    }
}

/// Point-wise Kleene logical _and_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn and_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.clone().binary(rhs.clone(), Operator::And)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn or_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.clone().binary(rhs.clone(), Operator::Or)
}

/// Execute a Kleene boolean operation between two arrays.
///
/// This is the entry point for boolean operations from the binary expression.
/// Handles constants and canonical boolean arrays directly, otherwise falls back to Arrow.
pub(crate) fn execute_boolean(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = boolean_nullability(lhs, rhs);

    if lhs.is_empty() {
        return Ok(Canonical::empty(&DType::Bool(nullable)).into_array());
    }

    if let Some(result) = constant_boolean(lhs, rhs, op)? {
        return Ok(result);
    }

    if let Some(lhs) = lhs.as_opt::<Bool>()
        && let Some(result) = <Bool as BooleanKernel>::boolean(lhs, rhs, op, ctx)?
    {
        return Ok(result);
    }

    if let Some(rhs) = rhs.as_opt::<Bool>()
        && let Some(result) = <Bool as BooleanKernel>::boolean(rhs, lhs, op, ctx)?
    {
        return Ok(result);
    }

    arrow_execute_boolean(lhs.clone(), rhs.clone(), op, ctx)
}

/// Arrow implementation for Kleene boolean operations using [`Operator`].
fn arrow_execute_boolean(
    lhs: ArrayRef,
    rhs: ArrayRef,
    op: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = boolean_nullability(&lhs, &rhs);
    let session = ctx.session().clone();

    let lhs = session
        .arrow()
        .execute_arrow(lhs, None, ctx)?
        .as_boolean_opt()
        .ok_or_else(|| vortex_err!("expected lhs to be boolean"))?
        .clone();

    let rhs = session
        .arrow()
        .execute_arrow(rhs, None, ctx)?
        .as_boolean_opt()
        .ok_or_else(|| vortex_err!("expected rhs to be boolean"))?
        .clone();

    let array = match op {
        Operator::And => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        Operator::Or => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
        other => vortex_bail!("Not a boolean operator: {other}"),
    };

    ArrayRef::from_arrow(&array, nullable == Nullability::Nullable)
}

/// Handles boolean operations where at least one operand is a constant array.
fn constant_boolean(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: Operator,
) -> VortexResult<Option<ArrayRef>> {
    let nullable = boolean_nullability(lhs, rhs);

    match (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>()) {
        (Some(lhs), Some(rhs)) => {
            let result = boolean_scalar_scalar(
                bool_scalar_value(lhs.scalar())?,
                bool_scalar_value(rhs.scalar())?,
                op,
            )?;

            Ok(Some(constant_bool_result(result, lhs.len(), nullable)))
        }
        (Some(lhs), None) => constant_array_boolean(lhs.scalar(), rhs, op, nullable),
        (None, Some(rhs)) => constant_array_boolean(rhs.scalar(), lhs, op, nullable),
        (None, None) => Ok(None),
    }
}

fn constant_array_boolean(
    constant: &Scalar,
    array: &ArrayRef,
    op: Operator,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    match (op, bool_scalar_value(constant)?) {
        (Operator::And, Some(false)) => Ok(Some(constant_bool_result(
            Some(false),
            array.len(),
            nullability,
        ))),
        (Operator::And, Some(true)) => Ok(Some(cast_bool_nullability(array, nullability)?)),
        (Operator::Or, Some(true)) => Ok(Some(constant_bool_result(
            Some(true),
            array.len(),
            nullability,
        ))),
        (Operator::Or, Some(false)) => Ok(Some(cast_bool_nullability(array, nullability)?)),
        (Operator::And | Operator::Or, None) => Ok(None),
        (other, _) => vortex_bail!("Not a boolean operator: {other}"),
    }
}

fn boolean_scalar_scalar(
    lhs: Option<bool>,
    rhs: Option<bool>,
    op: Operator,
) -> VortexResult<Option<bool>> {
    Ok(match op {
        Operator::And => match (lhs, rhs) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l & r),
        },
        Operator::Or => match (lhs, rhs) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l | r),
        },
        other => vortex_bail!("Not a boolean operator: {other}"),
    })
}

fn bool_scalar_value(scalar: &Scalar) -> VortexResult<Option<bool>> {
    Ok(scalar
        .as_bool_opt()
        .ok_or_else(|| vortex_err!("expected boolean scalar"))?
        .value())
}

/// Execute a Kleene boolean operation from boolean value bitmaps and validity values.
pub fn kleene_boolean_buffers(
    lhs_values: BitBuffer,
    lhs_validity: Validity,
    rhs_values: BitBuffer,
    rhs_validity: Validity,
    operator: Operator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs_values.len();
    debug_assert_eq!(rhs_values.len(), len);

    if lhs_validity.definitely_no_nulls() && rhs_validity.definitely_no_nulls() {
        let values = match operator {
            Operator::And => lhs_values & &rhs_values,
            Operator::Or => lhs_values | &rhs_values,
            other => vortex_bail!("Not a boolean operator: {other}"),
        };
        return Ok(BoolArray::try_new(values, Validity::from(nullability))?.into_array());
    }

    let lhs_valid = lhs_validity.execute_mask(len, ctx)?;
    let rhs_valid = rhs_validity.execute_mask(len, ctx)?;
    fused_boolean_buffers(
        len,
        &lhs_values,
        &lhs_valid,
        &rhs_values,
        &rhs_valid,
        operator,
        nullability,
    )
}

/// Execute a Kleene boolean operation between boolean value bits and a scalar.
pub fn kleene_boolean_buffer_scalar(
    values: BitBuffer,
    validity: Validity,
    scalar: &BoolScalar<'_>,
    operator: Operator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let scalar_value = scalar.value();
    let len = values.len();
    let result = match (operator, scalar_value) {
        (Operator::And, Some(false)) => {
            return Ok(constant_bool_result(Some(false), len, nullability));
        }
        (Operator::And, Some(true)) => {
            return Ok(
                BoolArray::try_new(values, validity.union_nullability(nullability))?.into_array(),
            );
        }
        (Operator::Or, Some(true)) => {
            return Ok(constant_bool_result(Some(true), len, nullability));
        }
        (Operator::Or, Some(false)) => {
            return Ok(
                BoolArray::try_new(values, validity.union_nullability(nullability))?.into_array(),
            );
        }
        (Operator::And, None) => {
            let valid = validity
                .execute_mask(len, ctx)?
                .bitand_not(&Mask::from_buffer(values));
            BoolArray::try_new(
                BitBuffer::new_unset(len),
                Validity::from_mask(valid, nullability),
            )?
        }
        (Operator::Or, None) => {
            let valid = validity.execute_mask(len, ctx)? & &Mask::from_buffer(values);
            BoolArray::try_new(
                BitBuffer::new_set(len),
                Validity::from_mask(valid, nullability),
            )?
        }
        (other, _) => vortex_bail!("Not a boolean operator: {other}"),
    };

    Ok(result.into_array())
}

fn fused_boolean_buffers(
    len: usize,
    lhs_values: &BitBuffer,
    lhs_validity: &Mask,
    rhs_values: &BitBuffer,
    rhs_validity: &Mask,
    operator: Operator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    if let Some(result) = fused_boolean_buffers_aligned(
        len,
        lhs_values,
        lhs_validity,
        rhs_values,
        rhs_validity,
        operator,
        nullability,
    )? {
        return Ok(result);
    }

    let n_words = len.div_ceil(64);

    macro_rules! fuse {
        ($lhs_valid_words:expr, $rhs_valid_words:expr) => {
            fused_boolean_words(
                len,
                lhs_values.chunks().iter_padded(),
                rhs_values.chunks().iter_padded(),
                $lhs_valid_words,
                $rhs_valid_words,
                operator,
                nullability,
            )
        };
    }

    match (lhs_validity.bit_buffer(), rhs_validity.bit_buffer()) {
        (AllOr::All, AllOr::All) => {
            fuse!(repeat_n(u64::MAX, n_words), repeat_n(u64::MAX, n_words))
        }
        (AllOr::All, AllOr::None) => {
            fuse!(repeat_n(u64::MAX, n_words), repeat_n(0, n_words))
        }
        (AllOr::All, AllOr::Some(rhs_validity)) => fuse!(
            repeat_n(u64::MAX, n_words),
            rhs_validity.chunks().iter_padded()
        ),
        (AllOr::None, AllOr::All) => {
            fuse!(repeat_n(0, n_words), repeat_n(u64::MAX, n_words))
        }
        (AllOr::None, AllOr::None) => {
            fuse!(repeat_n(0, n_words), repeat_n(0, n_words))
        }
        (AllOr::None, AllOr::Some(rhs_validity)) => {
            fuse!(repeat_n(0, n_words), rhs_validity.chunks().iter_padded())
        }
        (AllOr::Some(lhs_validity), AllOr::All) => fuse!(
            lhs_validity.chunks().iter_padded(),
            repeat_n(u64::MAX, n_words)
        ),
        (AllOr::Some(lhs_validity), AllOr::None) => {
            fuse!(lhs_validity.chunks().iter_padded(), repeat_n(0, n_words))
        }
        (AllOr::Some(lhs_validity), AllOr::Some(rhs_validity)) => fuse!(
            lhs_validity.chunks().iter_padded(),
            rhs_validity.chunks().iter_padded()
        ),
    }
}

#[derive(Clone, Copy)]
enum WordSource<'a> {
    Fill(u64),
    Bytes(&'a [u8]),
}

impl WordSource<'_> {
    #[inline]
    fn word_at(self, byte_offset: usize, len: usize) -> u64 {
        match self {
            Self::Fill(word) => word,
            Self::Bytes(bytes) => read_u64_le(&bytes[byte_offset..byte_offset + len]),
        }
    }
}

fn fused_boolean_buffers_aligned(
    len: usize,
    lhs_values: &BitBuffer,
    lhs_validity: &Mask,
    rhs_values: &BitBuffer,
    rhs_validity: &Mask,
    operator: Operator,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let Some(lhs_values) = word_source_from_bit_buffer(lhs_values) else {
        return Ok(None);
    };
    let Some(rhs_values) = word_source_from_bit_buffer(rhs_values) else {
        return Ok(None);
    };
    let Some(lhs_validity) = word_source_from_mask(lhs_validity) else {
        return Ok(None);
    };
    let Some(rhs_validity) = word_source_from_mask(rhs_validity) else {
        return Ok(None);
    };

    Ok(Some(fused_boolean_word_sources(
        len,
        lhs_values,
        rhs_values,
        lhs_validity,
        rhs_validity,
        operator,
        nullability,
    )?))
}

fn word_source_from_bit_buffer(buffer: &BitBuffer) -> Option<WordSource<'_>> {
    buffer.byte_aligned_bytes().map(WordSource::Bytes)
}

fn word_source_from_mask(mask: &Mask) -> Option<WordSource<'_>> {
    match mask.bit_buffer() {
        AllOr::All => Some(WordSource::Fill(u64::MAX)),
        AllOr::None => Some(WordSource::Fill(0)),
        AllOr::Some(buffer) => word_source_from_bit_buffer(buffer),
    }
}

fn fused_boolean_word_sources(
    len: usize,
    lhs_words: WordSource<'_>,
    rhs_words: WordSource<'_>,
    lhs_valid_words: WordSource<'_>,
    rhs_valid_words: WordSource<'_>,
    operator: Operator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    match operator {
        Operator::And => fused_boolean_and_word_sources(
            len,
            lhs_words,
            rhs_words,
            lhs_valid_words,
            rhs_valid_words,
            nullability,
        ),
        Operator::Or => fused_boolean_or_word_sources(
            len,
            lhs_words,
            rhs_words,
            lhs_valid_words,
            rhs_valid_words,
            nullability,
        ),
        other => vortex_bail!("Not a boolean operator: {other}"),
    }
}

fn fused_boolean_and_word_sources(
    len: usize,
    lhs_words: WordSource<'_>,
    rhs_words: WordSource<'_>,
    lhs_valid_words: WordSource<'_>,
    rhs_valid_words: WordSource<'_>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let n_bytes = len.div_ceil(8);
    let n_words = n_bytes.div_ceil(8);
    let full_bytes = n_bytes - n_bytes % 8;
    let mut values = BufferMut::<u64>::with_capacity(n_words);
    let mut validity = BufferMut::<u64>::with_capacity(n_words);

    for byte_offset in (0..full_bytes).step_by(8) {
        let lhs = lhs_words.word_at(byte_offset, 8);
        let rhs = rhs_words.word_at(byte_offset, 8);
        let lhs_valid = lhs_valid_words.word_at(byte_offset, 8);
        let rhs_valid = rhs_valid_words.word_at(byte_offset, 8);

        // SAFETY: both buffers were allocated with exactly `n_words` capacity, and this
        // loop plus the optional tail push emits at most `n_words` words.
        unsafe {
            values.push_unchecked(lhs & rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & !lhs) | (rhs_valid & !rhs));
        }
    }

    if full_bytes != n_bytes {
        let tail_len = n_bytes - full_bytes;
        let lhs = lhs_words.word_at(full_bytes, tail_len);
        let rhs = rhs_words.word_at(full_bytes, tail_len);
        let lhs_valid = lhs_valid_words.word_at(full_bytes, tail_len);
        let rhs_valid = rhs_valid_words.word_at(full_bytes, tail_len);

        // SAFETY: see the loop safety comment above.
        unsafe {
            values.push_unchecked(lhs & rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & !lhs) | (rhs_valid & !rhs));
        }
    }

    finish_fused_boolean_words(len, n_bytes, values, validity, nullability)
}

fn fused_boolean_or_word_sources(
    len: usize,
    lhs_words: WordSource<'_>,
    rhs_words: WordSource<'_>,
    lhs_valid_words: WordSource<'_>,
    rhs_valid_words: WordSource<'_>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let n_bytes = len.div_ceil(8);
    let n_words = n_bytes.div_ceil(8);
    let full_bytes = n_bytes - n_bytes % 8;
    let mut values = BufferMut::<u64>::with_capacity(n_words);
    let mut validity = BufferMut::<u64>::with_capacity(n_words);

    for byte_offset in (0..full_bytes).step_by(8) {
        let lhs = lhs_words.word_at(byte_offset, 8);
        let rhs = rhs_words.word_at(byte_offset, 8);
        let lhs_valid = lhs_valid_words.word_at(byte_offset, 8);
        let rhs_valid = rhs_valid_words.word_at(byte_offset, 8);

        // SAFETY: both buffers were allocated with exactly `n_words` capacity, and this
        // loop plus the optional tail push emits at most `n_words` words.
        unsafe {
            values.push_unchecked(lhs | rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & lhs) | (rhs_valid & rhs));
        }
    }

    if full_bytes != n_bytes {
        let tail_len = n_bytes - full_bytes;
        let lhs = lhs_words.word_at(full_bytes, tail_len);
        let rhs = rhs_words.word_at(full_bytes, tail_len);
        let lhs_valid = lhs_valid_words.word_at(full_bytes, tail_len);
        let rhs_valid = rhs_valid_words.word_at(full_bytes, tail_len);

        // SAFETY: see the loop safety comment above.
        unsafe {
            values.push_unchecked(lhs | rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & lhs) | (rhs_valid & rhs));
        }
    }

    finish_fused_boolean_words(len, n_bytes, values, validity, nullability)
}

fn finish_fused_boolean_words(
    len: usize,
    n_bytes: usize,
    values: BufferMut<u64>,
    validity: BufferMut<u64>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let mut values = values.into_byte_buffer();
    values.truncate(n_bytes);
    let mut validity = validity.into_byte_buffer();
    validity.truncate(n_bytes);
    Ok(BoolArray::try_new(
        BitBuffer::new(values.freeze(), len),
        Validity::from_mask(
            Mask::from_buffer(BitBuffer::new(validity.freeze(), len)),
            nullability,
        ),
    )?
    .into_array())
}

fn fused_boolean_words<L, R, LV, RV>(
    len: usize,
    lhs_words: L,
    rhs_words: R,
    lhs_valid_words: LV,
    rhs_valid_words: RV,
    operator: Operator,
    nullability: Nullability,
) -> VortexResult<ArrayRef>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
    LV: Iterator<Item = u64>,
    RV: Iterator<Item = u64>,
{
    match operator {
        Operator::And => fused_boolean_and_words(
            len,
            lhs_words,
            rhs_words,
            lhs_valid_words,
            rhs_valid_words,
            nullability,
        ),
        Operator::Or => fused_boolean_or_words(
            len,
            lhs_words,
            rhs_words,
            lhs_valid_words,
            rhs_valid_words,
            nullability,
        ),
        other => vortex_bail!("Not a boolean operator: {other}"),
    }
}

fn fused_boolean_and_words<L, R, LV, RV>(
    len: usize,
    lhs_words: L,
    rhs_words: R,
    lhs_valid_words: LV,
    rhs_valid_words: RV,
    nullability: Nullability,
) -> VortexResult<ArrayRef>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
    LV: Iterator<Item = u64>,
    RV: Iterator<Item = u64>,
{
    let n_words = len.div_ceil(64);
    let mut values = BufferMut::<u64>::with_capacity(n_words);
    let mut validity = BufferMut::<u64>::with_capacity(n_words);

    for (((lhs, rhs), lhs_valid), rhs_valid) in lhs_words
        .zip(rhs_words)
        .zip(lhs_valid_words)
        .zip(rhs_valid_words)
        .take(n_words)
    {
        // SAFETY: both buffers were allocated with exactly `n_words` capacity, and this loop is
        // capped at `n_words`.
        unsafe {
            values.push_unchecked(lhs & rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & !lhs) | (rhs_valid & !rhs));
        }
    }

    finish_fused_boolean_words(len, len.div_ceil(8), values, validity, nullability)
}

fn fused_boolean_or_words<L, R, LV, RV>(
    len: usize,
    lhs_words: L,
    rhs_words: R,
    lhs_valid_words: LV,
    rhs_valid_words: RV,
    nullability: Nullability,
) -> VortexResult<ArrayRef>
where
    L: Iterator<Item = u64>,
    R: Iterator<Item = u64>,
    LV: Iterator<Item = u64>,
    RV: Iterator<Item = u64>,
{
    let n_words = len.div_ceil(64);
    let mut values = BufferMut::<u64>::with_capacity(n_words);
    let mut validity = BufferMut::<u64>::with_capacity(n_words);

    for (((lhs, rhs), lhs_valid), rhs_valid) in lhs_words
        .zip(rhs_words)
        .zip(lhs_valid_words)
        .zip(rhs_valid_words)
        .take(n_words)
    {
        // SAFETY: both buffers were allocated with exactly `n_words` capacity, and this loop is
        // capped at `n_words`.
        unsafe {
            values.push_unchecked(lhs | rhs);
            validity
                .push_unchecked((lhs_valid & rhs_valid) | (lhs_valid & lhs) | (rhs_valid & rhs));
        }
    }

    finish_fused_boolean_words(len, len.div_ceil(8), values, validity, nullability)
}

fn constant_bool_result(value: Option<bool>, len: usize, nullability: Nullability) -> ArrayRef {
    let scalar = value
        .map(|b| Scalar::bool(b, nullability))
        .unwrap_or_else(|| Scalar::null(DType::Bool(nullability)));

    ConstantArray::new(scalar, len).into_array()
}

fn cast_bool_nullability(array: &ArrayRef, nullability: Nullability) -> VortexResult<ArrayRef> {
    let dtype = DType::Bool(nullability);
    if array.dtype() == &dtype {
        Ok(array.clone())
    } else {
        array.cast(dtype)
    }
}

fn boolean_nullability(lhs: &ArrayRef, rhs: &ArrayRef) -> Nullability {
    lhs.dtype().nullability() | rhs.dtype().nullability()
}

#[inline]
fn is_boolean_operator(operator: Operator) -> bool {
    matches!(operator, Operator::And | Operator::Or)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_kleene_truth_table() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = BoolArray::from_iter([
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ])
        .into_array();
        let rhs = BoolArray::from_iter([
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ])
        .into_array();

        assert_arrays_eq!(
            lhs.binary(rhs.clone(), Operator::And)?,
            BoolArray::from_iter([
                Some(true),
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                None,
                Some(false),
                None,
            ]),
            &mut ctx
        );

        assert_arrays_eq!(
            lhs.binary(rhs, Operator::Or)?,
            BoolArray::from_iter([
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                None,
                Some(true),
                None,
                None,
            ]),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_null_constant_kleene() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = BoolArray::from_iter([Some(false), Some(true), None]).into_array();
        let null = ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), lhs.len())
            .into_array();

        assert_arrays_eq!(
            lhs.binary(null.clone(), Operator::And)?,
            BoolArray::from_iter([Some(false), None, None]),
            &mut ctx
        );
        assert_arrays_eq!(
            lhs.binary(null, Operator::Or)?,
            BoolArray::from_iter([None, Some(true), None]),
            &mut ctx
        );

        Ok(())
    }

    #[rstest]
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    #[case(
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
    )]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = lhs.binary(rhs, Operator::Or).unwrap();
        #[expect(deprecated)]
        let r = r.to_bool().into_array();

        let v0 = r
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v1 = r
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v2 = r
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v3 = r
            .execute_scalar(3, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[rstest]
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    #[case(
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
    )]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        #[expect(deprecated)]
        let r = lhs
            .binary(rhs, Operator::And)
            .unwrap()
            .to_bool()
            .into_array();

        let v0 = r
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v1 = r
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v2 = r
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();
        let v3 = r
            .execute_scalar(3, &mut array_session().create_execution_ctx())
            .unwrap()
            .as_bool()
            .value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }
}
