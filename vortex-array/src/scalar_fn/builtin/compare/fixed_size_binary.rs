// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of fixed-size binary arrays, ordering values lexicographically by byte.
//!
//! Both operands share a byte width. Power-of-two widths reinterpret the values as `[u8; W]`
//! records, whose derived ordering is lexicographic, and reuse the lane comparison kernels. Other
//! widths compare one byte slice per row.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_compute::lane_kernels::collect_bits;
use vortex_compute::lane_kernels::collect_zip_bits;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::dtype::Nullability;
use crate::match_each_power_of_two_width;
use crate::scalar_fn::builtin::compare::compare_validity;
use crate::scalar_fn::builtin::compare::ordering_predicate;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::validity::Validity;

enum FixedSizeBinaryOperand {
    Array {
        values: ByteBuffer,
        validity: Validity,
    },
    Constant {
        value: ByteBuffer,
        validity: Validity,
    },
}

impl FixedSizeBinaryOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            let value = constant
                .scalar()
                .as_binary()
                .value()
                .cloned()
                .ok_or_else(|| vortex_err!("null constant handled by execute_compare"))?;
            return Ok(Self::Constant {
                value,
                validity: if constant.scalar().dtype().is_nullable() {
                    Validity::AllValid
                } else {
                    Validity::NonNullable
                },
            });
        }

        let array = array.clone().execute::<FixedSizeBinaryArray>(ctx)?;
        let validity = array.validity()?;
        Ok(Self::Array {
            values: array.values(),
            validity,
        })
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

/// Compare two fixed-size binary arrays of `byte_width` bytes per value.
pub(super) fn compare_fixed_size_binary(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    byte_width: u32,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let width = byte_width as usize;
    let lhs = FixedSizeBinaryOperand::try_new(lhs, ctx)?;
    let rhs = FixedSizeBinaryOperand::try_new(rhs, ctx)?;
    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;
    let allocator = ctx.allocator();

    let bits = match (lhs, rhs) {
        (
            FixedSizeBinaryOperand::Array { values: l, .. },
            FixedSizeBinaryOperand::Array { values: r, .. },
        ) => compare_values(l, r, width, len, op, allocator),
        (
            FixedSizeBinaryOperand::Array { values, .. },
            FixedSizeBinaryOperand::Constant { value, .. },
        ) => compare_constant(values, &value, width, len, op, allocator),
        (
            FixedSizeBinaryOperand::Constant { value, .. },
            FixedSizeBinaryOperand::Array { values, .. },
        ) => compare_constant(values, &value, width, len, op.swap(), allocator),
        (
            FixedSizeBinaryOperand::Constant { value: l, .. },
            FixedSizeBinaryOperand::Constant { value: r, .. },
        ) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full_in(
                ordering_predicate(op)(l.as_slice().cmp(r.as_slice())),
                len,
                allocator.clone(),
            )
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_values(
    lhs: ByteBuffer,
    rhs: ByteBuffer,
    width: usize,
    len: usize,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match_each_power_of_two_width!(
        width,
        |W| {
            compare_records(
                &Buffer::<[u8; W]>::from_byte_buffer(lhs),
                &Buffer::<[u8; W]>::from_byte_buffer(rhs),
                op,
                allocator,
            )
        },
        _ => { compare_slices(&lhs, &rhs, width, len, op, allocator) }
    )
}

fn compare_constant(
    values: ByteBuffer,
    constant: &[u8],
    width: usize,
    len: usize,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match_each_power_of_two_width!(
        width,
        |W| {
            compare_records_constant(
                &Buffer::<[u8; W]>::from_byte_buffer(values),
                constant,
                op,
                allocator,
            )
        },
        _ => { compare_slices_constant(&values, constant, width, len, op, allocator) }
    )
}

/// Compares records pairwise with the lane kernels.
fn compare_records<const W: usize>(
    lhs: &[[u8; W]],
    rhs: &[[u8; W]],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_zip_bits(lhs, rhs, |a, b| a == b, allocator),
        CompareOperator::NotEq => collect_zip_bits(lhs, rhs, |a, b| a != b, allocator),
        CompareOperator::Gt => collect_zip_bits(lhs, rhs, |a, b| a > b, allocator),
        CompareOperator::Gte => collect_zip_bits(lhs, rhs, |a, b| a >= b, allocator),
        CompareOperator::Lt => collect_zip_bits(lhs, rhs, |a, b| a < b, allocator),
        CompareOperator::Lte => collect_zip_bits(lhs, rhs, |a, b| a <= b, allocator),
    }
}

/// Compares every record against `constant`, which has `W` bytes.
fn compare_records_constant<const W: usize>(
    values: &[[u8; W]],
    constant: &[u8],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let mut c = [0u8; W];
    c.copy_from_slice(constant);
    match op {
        CompareOperator::Eq => collect_bits(values, |a| a == c, allocator),
        CompareOperator::NotEq => collect_bits(values, |a| a != c, allocator),
        CompareOperator::Gt => collect_bits(values, |a| a > c, allocator),
        CompareOperator::Gte => collect_bits(values, |a| a >= c, allocator),
        CompareOperator::Lt => collect_bits(values, |a| a < c, allocator),
        CompareOperator::Lte => collect_bits(values, |a| a <= c, allocator),
    }
}

/// Compares `len` rows of `width` bytes pairwise.
fn compare_slices(
    lhs: &[u8],
    rhs: &[u8],
    width: usize,
    len: usize,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let predicate = ordering_predicate(op);
    BitBuffer::collect_bool_in(
        len,
        |i| predicate(row(lhs, width, i).cmp(row(rhs, width, i))),
        allocator.clone(),
    )
}

/// Compares `len` rows of `width` bytes against `constant`.
fn compare_slices_constant(
    values: &[u8],
    constant: &[u8],
    width: usize,
    len: usize,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let predicate = ordering_predicate(op);
    BitBuffer::collect_bool_in(
        len,
        |i| predicate(row(values, width, i).cmp(constant)),
        allocator.clone(),
    )
}

#[inline]
fn row(values: &[u8], width: usize, index: usize) -> &[u8] {
    &values[index * width..(index + 1) * width]
}
