// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::BlockedFoR;
use crate::BlockedFoRArray;
use crate::BlockedFoRData;
use crate::blocked_for::array::BLOCK_SIZE;

impl BlockedFoRData {
    /// Encode a primitive array, subtracting a per-block minimum from every block of
    /// [`BLOCK_SIZE`] values.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<BlockedFoRArray> {
        let validity = array.validity()?;
        let (encoded, references) = match_each_integer_ptype!(array.ptype(), |T| {
            let (residuals, references) = compress_primitive::<T>(&array, ctx)?;
            (
                PrimitiveArray::new(residuals, validity).into_array(),
                PrimitiveArray::new(references, Validity::NonNullable).into_array(),
            )
        });
        BlockedFoR::try_new(encoded, references, 0)
    }
}

/// Split `array` into [`BLOCK_SIZE`] blocks, returning the block-local residuals and the
/// per-block reference (minimum) values.
///
/// Null values encode as a zero residual, exactly as [`crate::FoR`] does, so that they decode
/// back to their block's reference rather than wrapping out of the primitive range.
fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Buffer<T>, Buffer<T>)> {
    let len = array.len();
    let values = array.as_slice::<T>();
    let num_blocks = len.div_ceil(BLOCK_SIZE);

    let mut references = BufferMut::<T>::with_capacity(num_blocks);
    let mut residuals = BufferMut::<T>::with_capacity(len);

    match array.validity()?.execute_mask(len, ctx)?.bit_buffer() {
        AllOr::All => {
            for block in values.chunks(BLOCK_SIZE) {
                let min = block.iter().copied().min().unwrap_or_else(T::zero);
                references.push(min);
                residuals.extend(block.iter().map(|v| v.wrapping_sub(&min)));
            }
        }
        AllOr::None => {
            references.extend(std::iter::repeat_n(T::zero(), num_blocks));
            residuals.extend(std::iter::repeat_n(T::zero(), len));
        }
        AllOr::Some(bits) => {
            // Materialize the validity bits once: the per-block loops below need random access
            // to them, and `BitBuffer::value` is far too costly to call per element.
            let valid = bits.iter().collect::<Vec<_>>();
            for (block, valid) in values.chunks(BLOCK_SIZE).zip(valid.chunks(BLOCK_SIZE)) {
                let min = block
                    .iter()
                    .zip(valid)
                    .filter_map(|(v, valid)| valid.then_some(*v))
                    .min()
                    .unwrap_or_else(T::zero);
                references.push(min);
                residuals.extend(block.iter().zip(valid).map(|(v, valid)| {
                    if *valid {
                        v.wrapping_sub(&min)
                    } else {
                        T::zero()
                    }
                }));
            }
        }
    }

    Ok((residuals.freeze(), references.freeze()))
}

/// Per-block summary used to estimate how well blocked FoR will pack.
pub struct BlockSummary {
    /// The widest `max - min` over the valid values of any block.
    pub max_range: u128,
    /// Whether every block's minimum is already zero, making the encoding a no-op.
    pub all_minima_zero: bool,
}

/// Summarize `array` a block at a time, or `None` if it holds no valid values.
pub fn block_summary(
    array: ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> Option<BlockSummary> {
    let len = array.len();
    let validity = array.validity().ok()?.execute_mask(len, exec_ctx).ok()?;

    match_each_integer_ptype!(array.ptype(), |T| {
        block_summary_typed::<T>(array.as_slice::<T>(), validity.bit_buffer())
    })
}

fn block_summary_typed<T: NativePType + PrimInt>(
    values: &[T],
    validity: AllOr<&BitBuffer>,
) -> Option<BlockSummary> {
    // Materialize the validity bits once rather than probing them per element.
    let valid = match validity {
        AllOr::All => None,
        AllOr::None => return None,
        AllOr::Some(bits) => Some(bits.iter().collect::<Vec<_>>()),
    };

    let mut max_range = 0u128;
    let mut all_minima_zero = true;
    let mut any_valid = false;

    for (index, block) in values.chunks(BLOCK_SIZE).enumerate() {
        let (min, max) = match &valid {
            None => (block.iter().copied().min(), block.iter().copied().max()),
            Some(valid) => {
                let valid = &valid[index * BLOCK_SIZE..][..block.len()];
                let mut it = block
                    .iter()
                    .zip(valid)
                    .filter_map(|(v, valid)| valid.then_some(*v));
                match it.next() {
                    None => (None, None),
                    Some(first) => {
                        let (min, max) =
                            it.fold((first, first), |(min, max), v| (min.min(v), max.max(v)));
                        (Some(min), Some(max))
                    }
                }
            }
        };

        let (Some(min), Some(max)) = (min, max) else {
            // An all-null block encodes to a zero reference and zero residuals.
            continue;
        };
        any_valid = true;
        all_minima_zero &= min.is_zero();
        let range = max.to_i128()? - min.to_i128()?;
        #[expect(
            clippy::cast_sign_loss,
            reason = "max >= min, so the range is non-negative"
        )]
        let range = range as u128;
        max_range = max_range.max(range);
    }

    any_valid.then_some(BlockSummary {
        max_range,
        all_minima_zero,
    })
}
