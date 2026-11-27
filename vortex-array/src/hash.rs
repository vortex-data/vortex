// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::patches::Patches;
use crate::validity::Validity;

/// The equality mode for structural equality and hashing of arrays.
///
/// This configuration option defines how precise the hash/equals results are for the set of
/// data buffers backing the array.
#[derive(Clone, Copy, Debug)]
pub enum EqMode {
    /// Data buffers are compared by their pointer and length only. This is the fastest option, but
    /// may lead to false negatives if two arrays contain identical data but are backed by
    /// different buffers.
    Ptr,
    /// Data buffers are compared by their full content. This is the slowest option, but guarantees
    /// that two arrays with identical data will be considered equal.
    Value,
}

/// A hash trait for arrays that represents structural equality with a configurable equality mode.
/// This trait is used primarily to implement common subtree elimination and other
/// array-based caching mechanisms.
///
/// The equality mode defines what level of structural equality is represented. See
/// [`EqMode`] for more details.
///
/// Note that where [`EqMode::Ptr`] is used, the hash is only valid for the lifetime of the
/// object.
pub trait ArrayHash {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode);
}

/// A dynamic version of [`ArrayHash`].
pub trait DynArrayDataHash: private::SealedHash {
    fn dyn_array_hash(&self, state: &mut dyn Hasher, eq_mode: EqMode);
}

impl<T: ArrayHash + ?Sized> DynArrayDataHash for T {
    fn dyn_array_hash(&self, mut state: &mut dyn Hasher, eq_mode: EqMode) {
        ArrayHash::array_hash(self, &mut state, eq_mode);
    }
}

/// An equality trait for arrays that represents structural equality with a configurable equality
/// mode. This trait is used primarily to implement common subtree elimination and other
/// array-based caching mechanisms.
///
/// The equality mode defines what level of structural equality is represented. See
/// [`EqMode`] for more details.
pub trait ArrayEq {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool;
}

/// A dynamic version of [`ArrayEq`].
pub trait DynArrayDataEq: private::SealedEq {
    fn dyn_array_eq(&self, other: &dyn Any, eq_mode: EqMode) -> bool;
}

impl<T: ArrayEq + 'static> DynArrayDataEq for T {
    fn dyn_array_eq(&self, other: &dyn Any, eq_mode: EqMode) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| ArrayEq::array_eq(self, other, eq_mode))
    }
}

mod private {
    use crate::ArrayEq;
    use crate::ArrayHash;

    pub trait SealedHash {}
    impl<T: ArrayHash + ?Sized> SealedHash for T {}
    pub trait SealedEq {}
    impl<T: ArrayEq + ?Sized> SealedEq for T {}
}

impl<T: Hash> ArrayHash for Buffer<T> {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        match eq_mode {
            EqMode::Ptr => {
                self.as_ptr().hash(state);
                self.len().hash(state);
            }
            EqMode::Value => {
                self.as_ref().hash(state);
            }
        }
    }
}
impl<T: PartialEq> ArrayEq for Buffer<T> {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        match eq_mode {
            EqMode::Ptr => self.as_ptr() == other.as_ptr() && self.len() == other.len(),
            EqMode::Value => self.as_ref() == other.as_ref(),
        }
    }
}

impl ArrayHash for BitBuffer {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        match eq_mode {
            EqMode::Ptr => {
                self.inner().as_ptr().hash(state);
                self.offset().hash(state);
                self.len().hash(state);
            }
            EqMode::Value => {
                // NOTE(ngates): this is really rather expensive...
                for chunk in self.chunks().iter_padded() {
                    chunk.hash(state);
                }
            }
        }
    }
}
impl ArrayEq for BitBuffer {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        match eq_mode {
            EqMode::Ptr => {
                self.inner().as_ptr() == other.inner().as_ptr()
                    && self.offset() == other.offset()
                    && self.len() == other.len()
            }
            EqMode::Value => self.eq(other),
        }
    }
}

impl<T: ArrayHash> ArrayHash for Option<T> {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        match self {
            Some(value) => {
                true.hash(state);
                value.array_hash(state, eq_mode);
            }
            None => {
                false.hash(state);
            }
        }
    }
}

impl<T: ArrayEq> ArrayEq for Option<T> {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        match (self, other) {
            (Some(v1), Some(v2)) => v1.array_eq(v2, eq_mode),
            (None, None) => true,
            _ => false,
        }
    }
}

impl ArrayHash for Mask {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        std::mem::discriminant(self).hash(state);
        match self {
            Mask::AllTrue(len) => {
                len.hash(state);
            }
            Mask::AllFalse(len) => {
                len.hash(state);
            }
            Mask::Values(values) => {
                values.bit_buffer().array_hash(state, eq_mode);
            }
        }
    }
}
impl ArrayEq for Mask {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        match (self, other) {
            (Mask::AllTrue(len1), Mask::AllTrue(len2)) => len1 == len2,
            (Mask::AllFalse(len1), Mask::AllFalse(len2)) => len1 == len2,
            (Mask::Values(buf1), Mask::Values(buf2)) => {
                buf1.bit_buffer().array_eq(buf2.bit_buffer(), eq_mode)
            }
            _ => false,
        }
    }
}

impl ArrayHash for Validity {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        std::mem::discriminant(self).hash(state);
        if let Validity::Array(array) = self {
            array.array_hash(state, eq_mode);
        }
    }
}

impl ArrayEq for Validity {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        match (self, other) {
            (Validity::AllValid, Validity::AllValid) => true,
            (Validity::AllInvalid, Validity::AllInvalid) => true,
            (Validity::NonNullable, Validity::NonNullable) => true,
            (Validity::Array(arr1), Validity::Array(arr2)) => arr1.array_eq(arr2, eq_mode),
            _ => false,
        }
    }
}

impl ArrayHash for Patches {
    fn array_hash<H: Hasher>(&self, state: &mut H, eq_mode: EqMode) {
        self.array_len().hash(state);
        self.offset().hash(state);
        self.indices().array_hash(state, eq_mode);
        self.values().array_hash(state, eq_mode);
    }
}

impl ArrayEq for Patches {
    fn array_eq(&self, other: &Self, eq_mode: EqMode) -> bool {
        self.array_len() == other.array_len()
            && self.offset() == other.offset()
            && self.indices().array_eq(other.indices(), eq_mode)
            && self.values().array_eq(other.values(), eq_mode)
    }
}

#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;
    use std::hash::Hasher;
    use std::sync::LazyLock;

    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayEq;
    use crate::ArrayHash;
    use crate::DynArrayDataEq;
    use crate::DynArrayDataHash;
    use crate::EqMode;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::patches::Patches;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn hash_of<T: ArrayHash + ?Sized>(value: &T, eq_mode: EqMode) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.array_hash(&mut hasher, eq_mode);
        hasher.finish()
    }

    fn dyn_hash_of<T: DynArrayDataHash + ?Sized>(value: &T, eq_mode: EqMode) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.dyn_array_hash(&mut hasher, eq_mode);
        hasher.finish()
    }

    #[test]
    fn test_buffer_eq_and_hash_modes() {
        let buf_a: Buffer<i32> = buffer![1, 2, 3, 4];
        let buf_a_clone = buf_a.clone();
        let buf_b: Buffer<i32> = Buffer::copy_from([1, 2, 3, 4]);
        let buf_diff: Buffer<i32> = buffer![1, 2, 3, 5];
        let buf_prefix = buf_a.slice(0..3);

        // Same backing allocation: equal in both modes with matching hashes.
        assert!(buf_a.array_eq(&buf_a_clone, EqMode::Ptr));
        assert!(buf_a.array_eq(&buf_a_clone, EqMode::Value));
        assert_eq!(
            hash_of(&buf_a, EqMode::Ptr),
            hash_of(&buf_a_clone, EqMode::Ptr)
        );
        assert_eq!(
            hash_of(&buf_a, EqMode::Value),
            hash_of(&buf_a_clone, EqMode::Value)
        );

        // Distinct allocations with identical contents: unequal in Ptr, equal in Value.
        assert!(!buf_a.array_eq(&buf_b, EqMode::Ptr));
        assert!(buf_a.array_eq(&buf_b, EqMode::Value));
        assert_eq!(
            hash_of(&buf_a, EqMode::Value),
            hash_of(&buf_b, EqMode::Value)
        );

        // Same pointer but different length: unequal in both modes.
        assert!(!buf_a.array_eq(&buf_prefix, EqMode::Ptr));
        assert!(!buf_a.array_eq(&buf_prefix, EqMode::Value));

        // Different values: unequal in both modes.
        assert!(!buf_a.array_eq(&buf_diff, EqMode::Ptr));
        assert!(!buf_a.array_eq(&buf_diff, EqMode::Value));

        // Dynamic dispatch wrappers match concrete trait behavior and reject mismatched types.
        assert!(buf_a.dyn_array_eq(&buf_b, EqMode::Value));
        assert!(!buf_a.dyn_array_eq(&buf_b, EqMode::Ptr));
        assert!(!buf_a.dyn_array_eq(&42u32, EqMode::Value));
        assert_eq!(
            dyn_hash_of(&buf_a, EqMode::Value),
            hash_of(&buf_a, EqMode::Value)
        );

        // Option<T> delegates to inner ArrayEq/ArrayHash.
        let opt_a = Some(buf_a);
        let opt_b = Some(buf_b);
        let none: Option<Buffer<i32>> = None;
        assert!(none.array_eq(&None, EqMode::Ptr));
        assert_eq!(hash_of(&none, EqMode::Ptr), hash_of(&none, EqMode::Value));
        assert!(!opt_a.array_eq(&none, EqMode::Value));
        assert!(!opt_a.array_eq(&opt_b, EqMode::Ptr));
        assert!(opt_a.array_eq(&opt_b, EqMode::Value));
        assert_eq!(
            hash_of(&opt_a, EqMode::Value),
            hash_of(&opt_b, EqMode::Value)
        );
    }

    #[test]
    fn test_bit_buffer_eq_and_hash_unaligned_and_padded() {
        // Two bit buffers with identical logical bits (length 67, non-multiple of 64) where
        // one is a fresh aligned buffer and the other is sliced at an unaligned bit offset
        // with garbage padding bits outside the logical length.
        let pattern: Vec<bool> = (0..67).map(|i| i % 3 == 0 || i % 7 == 1).collect();
        let aligned = BitBuffer::from_iter(pattern.iter().copied());
        let aligned_clone = aligned.clone();

        let mut padded_bits = vec![true, false, true]; // 3-bit prefix offset
        padded_bits.extend_from_slice(&pattern);
        padded_bits.extend_from_slice(&[true, true, false, true, true]); // trailing junk bits
        let unaligned_backing = BitBuffer::from_iter(padded_bits);
        let unaligned = unaligned_backing.slice(3..3 + pattern.len());

        // Same allocation and same offset/len.
        assert!(aligned.array_eq(&aligned_clone, EqMode::Ptr));
        assert!(aligned.array_eq(&aligned_clone, EqMode::Value));
        assert_eq!(
            hash_of(&aligned, EqMode::Ptr),
            hash_of(&aligned_clone, EqMode::Ptr)
        );

        // Unaligned slice vs aligned buffer with same logical bits.
        assert!(!aligned.array_eq(&unaligned, EqMode::Ptr));
        assert!(aligned.array_eq(&unaligned, EqMode::Value));
        assert_eq!(
            hash_of(&aligned, EqMode::Value),
            hash_of(&unaligned, EqMode::Value)
        );

        // Two slices of the SAME backing buffer at different offsets that happen to hold the
        // same bits: Ptr mode distinguishes offsets, Value mode treats them as equal.
        let repeated = BitBuffer::from_iter([true, false, true, false, true, false, true, false]);
        let slice_a = repeated.slice(0..5);
        let slice_b = repeated.slice(2..7);
        assert!(!slice_a.array_eq(&slice_b, EqMode::Ptr));
        assert!(slice_a.array_eq(&slice_b, EqMode::Value));
        assert_eq!(
            hash_of(&slice_a, EqMode::Value),
            hash_of(&slice_b, EqMode::Value)
        );

        // Non-byte-aligned trailing bits in the backing byte buffer must be masked out in Value
        // hash/eq (0b0000_0101 vs 0b1111_0101 at len 4).
        let short_a = BitBuffer::new(ByteBuffer::copy_from([0b0000_0101u8]), 4);
        let short_b = BitBuffer::new(ByteBuffer::copy_from([0b1111_0101u8]), 4);
        assert!(!short_a.array_eq(&short_b, EqMode::Ptr));
        assert!(short_a.array_eq(&short_b, EqMode::Value));
        assert_eq!(
            hash_of(&short_a, EqMode::Value),
            hash_of(&short_b, EqMode::Value)
        );
    }

    #[test]
    fn test_mask_eq_and_hash() {
        let all_true = Mask::new_true(5);
        let all_false = Mask::new_false(5);

        assert!(all_true.array_eq(&Mask::new_true(5), EqMode::Ptr));
        assert!(!all_true.array_eq(&Mask::new_true(6), EqMode::Value));
        assert!(all_false.array_eq(&Mask::new_false(5), EqMode::Ptr));
        assert!(!all_false.array_eq(&Mask::new_false(6), EqMode::Value));
        assert!(!all_true.array_eq(&all_false, EqMode::Value));
        assert_eq!(
            hash_of(&all_true, EqMode::Value),
            hash_of(&Mask::new_true(5), EqMode::Value)
        );
        assert_eq!(
            hash_of(&all_false, EqMode::Value),
            hash_of(&Mask::new_false(5), EqMode::Value)
        );

        let bits_a = BitBuffer::from_iter([true, false, true, false, true]);
        let bits_b = BitBuffer::from_iter([false, true, false, true, false, true]).slice(1..6);
        let mask_a = Mask::from_buffer(bits_a);
        let mask_a_clone = mask_a.clone();
        let mask_b = Mask::from_buffer(bits_b);

        assert!(mask_a.array_eq(&mask_a_clone, EqMode::Ptr));
        assert!(!mask_a.array_eq(&mask_b, EqMode::Ptr));
        assert!(mask_a.array_eq(&mask_b, EqMode::Value));
        assert_eq!(
            hash_of(&mask_a, EqMode::Value),
            hash_of(&mask_b, EqMode::Value)
        );
        assert!(!mask_a.array_eq(&all_true, EqMode::Value));
        assert!(!mask_a.array_eq(&all_false, EqMode::Value));
    }

    #[test]
    fn test_validity_eq_and_hash() {
        let all_valid = Validity::AllValid;
        let all_invalid = Validity::AllInvalid;
        let non_nullable = Validity::NonNullable;

        for (lhs, rhs) in [
            (&all_valid, &Validity::AllValid),
            (&all_invalid, &Validity::AllInvalid),
            (&non_nullable, &Validity::NonNullable),
        ] {
            assert!(lhs.array_eq(rhs, EqMode::Ptr));
            assert!(lhs.array_eq(rhs, EqMode::Value));
            assert_eq!(hash_of(lhs, EqMode::Ptr), hash_of(rhs, EqMode::Ptr));
            assert_eq!(hash_of(lhs, EqMode::Value), hash_of(rhs, EqMode::Value));
        }
        assert!(!all_valid.array_eq(&all_invalid, EqMode::Value));
        assert!(!all_valid.array_eq(&non_nullable, EqMode::Value));
        assert!(!all_invalid.array_eq(&non_nullable, EqMode::Value));

        let bool_a = BoolArray::from_iter([true, false, true]).into_array();
        let bool_b = BoolArray::from_iter([true, false, true]).into_array();
        let val_a = Validity::Array(bool_a);
        let val_a_clone = val_a.clone();
        let val_b = Validity::Array(bool_b);

        assert!(val_a.array_eq(&val_a_clone, EqMode::Ptr));
        assert!(!val_a.array_eq(&val_b, EqMode::Ptr));
        assert!(val_a.array_eq(&val_b, EqMode::Value));
        assert_eq!(
            hash_of(&val_a, EqMode::Value),
            hash_of(&val_b, EqMode::Value)
        );
        assert!(!val_a.array_eq(&all_valid, EqMode::Value));
    }

    #[test]
    fn test_patches_eq_and_hash() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let indices_a = buffer![1u32, 4].into_array();
        let values_a = buffer![10i32, 40].into_array();
        let patches_a = Patches::new(10, 0, indices_a, values_a, None, &mut ctx)?;
        let patches_a_clone = patches_a.clone();

        let indices_b = Buffer::copy_from([1u32, 4]).into_array();
        let values_b = Buffer::copy_from([10i32, 40]).into_array();
        let patches_b = Patches::new(10, 0, indices_b, values_b, None, &mut ctx)?;

        // Same backing arrays: equal in both modes.
        assert!(patches_a.array_eq(&patches_a_clone, EqMode::Ptr));
        assert!(patches_a.array_eq(&patches_a_clone, EqMode::Value));
        assert_eq!(
            hash_of(&patches_a, EqMode::Ptr),
            hash_of(&patches_a_clone, EqMode::Ptr)
        );

        // Separately allocated identical indices/values: unequal in Ptr, equal in Value.
        assert!(!patches_a.array_eq(&patches_b, EqMode::Ptr));
        assert!(patches_a.array_eq(&patches_b, EqMode::Value));
        assert_eq!(
            hash_of(&patches_a, EqMode::Value),
            hash_of(&patches_b, EqMode::Value)
        );

        // Different array_len, offset, indices, or values are unequal in both modes.
        let diff_len = Patches::new(
            12,
            0,
            patches_a.indices().clone(),
            patches_a.values().clone(),
            None,
            &mut ctx,
        )?;
        assert!(!patches_a.array_eq(&diff_len, EqMode::Value));

        let diff_indices = Patches::new(
            10,
            0,
            buffer![1u32, 5].into_array(),
            patches_a.values().clone(),
            None,
            &mut ctx,
        )?;
        assert!(!patches_a.array_eq(&diff_indices, EqMode::Value));

        let diff_offset = Patches::new(
            10,
            1,
            buffer![2u32, 5].into_array(),
            patches_a.values().clone(),
            None,
            &mut ctx,
        )?;
        assert!(!patches_a.array_eq(&diff_offset, EqMode::Value));

        let diff_values = Patches::new(
            10,
            0,
            patches_a.indices().clone(),
            buffer![10i32, 99].into_array(),
            None,
            &mut ctx,
        )?;
        assert!(!patches_a.array_eq(&diff_values, EqMode::Value));

        Ok(())
    }
}
