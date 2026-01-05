// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Casting utilities for primitive vectors.

use num_traits::NumCast;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::match_each_signed_integer_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::PVector;
use super::PrimitiveVector;
use super::PrimitiveVectorMut;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::match_each_integer_pvector;
use crate::match_each_integer_pvector_mut;

/// Cast a [`PVector<Src>`] to a [`PVector<Dst>`] by converting each element.
///
/// # Errors
///
/// Returns an error if any valid element cannot be converted (e.g., overflow).
pub fn cast_pvector<Src: NativePType, Dst: NativePType>(
    src: &PVector<Src>,
) -> VortexResult<PVector<Dst>> {
    let elements: &[Src] = src.as_ref();
    match src.validity().bit_buffer() {
        AllOr::All => {
            let mut buffer = BufferMut::with_capacity(elements.len());
            for &item in elements {
                let converted = <Dst as NumCast>::from(item).ok_or_else(
                    || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, Dst::PTYPE),
                )?;
                // SAFETY: We pre-allocated the required capacity.
                unsafe { buffer.push_unchecked(converted) }
            }
            Ok(PVector::from(buffer.freeze()))
        }
        AllOr::None => Ok(PVector::new(
            Buffer::zeroed(elements.len()),
            Mask::new_false(elements.len()),
        )),
        AllOr::Some(bit_buffer) => {
            let mut buffer = BufferMut::with_capacity(elements.len());
            for (&item, valid) in elements.iter().zip(bit_buffer.iter()) {
                if valid {
                    let converted = <Dst as NumCast>::from(item).ok_or_else(
                        || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, Dst::PTYPE),
                    )?;
                    // SAFETY: We pre-allocated the required capacity.
                    unsafe { buffer.push_unchecked(converted) }
                } else {
                    // SAFETY: We pre-allocated the required capacity.
                    unsafe { buffer.push_unchecked(Dst::default()) }
                }
            }
            Ok(PVector::new(buffer.freeze(), src.validity().clone()))
        }
    }
}

impl PrimitiveVectorMut {
    /// Upcasts this integer vector to a wider integer type with matching signedness.
    ///
    /// Returns self unchanged if the target type is the same or smaller in byte width.
    #[expect(
        clippy::cognitive_complexity,
        reason = "complexity from nested match_each_* macros"
    )]
    pub fn upcast(self, target: PType) -> Self {
        debug_assert!(self.ptype().is_int());
        debug_assert!(target.is_int());
        debug_assert!(
            (self.ptype().is_signed_int() && target.is_signed_int())
                || (self.ptype().is_unsigned_int() && target.is_unsigned_int())
        );

        // No-op if already at target or target is same/smaller
        if self.ptype() == target || target.byte_width() <= self.ptype().byte_width() {
            return self;
        }

        // Freeze to immutable, cast, then convert back to mutable
        let frozen = self.freeze();
        match_each_integer_pvector!(&frozen, |src_vec| {
            if target.is_unsigned_int() {
                match_each_unsigned_integer_ptype!(target, |Dst| {
                    let casted = cast_pvector::<_, Dst>(src_vec)
                        .vortex_expect("upcast should never fail for widening casts");
                    casted.into_mut().into()
                })
            } else {
                match_each_signed_integer_ptype!(target, |Dst| {
                    let casted = cast_pvector::<_, Dst>(src_vec)
                        .vortex_expect("upcast should never fail for widening casts");
                    casted.into_mut().into()
                })
            }
        })
    }

    /// Extends this vector from another, automatically upcasting if needed.
    ///
    /// Unlike [`VectorMutOps::extend_from_vector`](VectorMutOps::extend_from_vector), this does
    /// **NOT** panic on type mismatch. Instead, it upcasts `self` to the wider of the two types.
    ///
    /// Returns the new [`PType`] after any upcasting.
    pub fn extend_from_vector_with_upcast(&mut self, other: &PrimitiveVector) -> PType {
        debug_assert!(self.ptype().is_int());
        debug_assert!(other.ptype().is_int());

        let target = self
            .ptype()
            .to_unsigned()
            .max_unsigned_ptype(other.ptype().to_unsigned());

        if self.ptype() != target {
            let old_self = std::mem::replace(self, Self::with_capacity(target, 0));
            *self = old_self.upcast(target);
        }

        // Now extend with casting from other
        self.reserve(other.len());
        extend_with_cast(self, other);
        self.ptype()
    }
}

/// Extends `dst` with values from `src`, casting each element.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from nested match_each_* macros"
)]
fn extend_with_cast(dst: &mut PrimitiveVectorMut, src: &PrimitiveVector) {
    match_each_integer_pvector_mut!(dst, |dst_vec| {
        match_each_integer_pvector!(src, |src_vec| {
            let src_slice = src_vec.as_ref();
            let src_validity = src_vec.validity();
            for i in 0..src_vec.len() {
                if src_validity.value(i) {
                    #[allow(clippy::unnecessary_cast)]
                    let converted = <_ as NumCast>::from(src_slice[i])
                        .vortex_expect("conversion should succeed after upcast");
                    dst_vec.push_opt(Some(converted));
                } else {
                    dst_vec.push_opt(None);
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PType;
    use vortex_dtype::PTypeDowncast;

    use super::*;
    use crate::primitive::PVectorMut;

    #[test]
    fn test_upcast_unsigned() {
        let mut vec: PrimitiveVectorMut =
            PVectorMut::<u8>::from_iter([0u8, u8::MAX].map(Some)).into();
        let other: PrimitiveVector = PVectorMut::<u32>::from_iter([u32::MAX].map(Some))
            .freeze()
            .into();

        vec.extend_from_vector_with_upcast(&other);
        assert_eq!(vec.ptype(), PType::U32);

        let frozen = vec.freeze().into_u32();
        assert_eq!(frozen.as_ref(), &[0, u8::MAX as u32, u32::MAX]);
    }

    #[test]
    fn test_upcast_signed() {
        let vec: PrimitiveVectorMut =
            PVectorMut::<i8>::from_iter([i8::MIN, i8::MAX].map(Some)).into();

        let mut vec = vec.upcast(PType::I32);
        let other: PrimitiveVector = PVectorMut::<i32>::from_iter([i32::MIN, i32::MAX].map(Some))
            .freeze()
            .into();
        extend_with_cast(&mut vec, &other);

        let frozen = vec.freeze().into_i32();
        assert_eq!(
            frozen.as_ref(),
            &[i8::MIN as i32, i8::MAX as i32, i32::MIN, i32::MAX]
        );
    }
}
