// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The dictionary shared by the encoders of fixed-width values.

use std::cell::OnceCell;
use std::hash::Hash;
use std::mem;

use rustc_hash::FxBuildHasher;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::UnsignedPType;

/// The dictionary of an encoder of fixed-width values `V`, looked up by a key `K` of each value.
///
/// Null values are stored in the values of the dictionary such that codes are always non-null.
pub(super) struct ValuesDict<K, V, Code> {
    lookup: HashMap<K, Code, FxBuildHasher>,
    null_code: OnceCell<Code>,
    values: BufferMut<V>,
    values_nulls: BitBufferMut,
    max_dict_len: usize,
    allocator: BufferAllocatorRef,
}

impl<K, V, Code> ValuesDict<K, V, Code>
where
    K: Hash + Eq,
    V: Copy + Default,
    Code: UnsignedPType,
{
    /// Creates an empty dictionary of at most `max_dict_len` values.
    pub(super) fn new_in(max_dict_len: usize, allocator: BufferAllocatorRef) -> Self {
        Self {
            lookup: HashMap::with_hasher(FxBuildHasher),
            null_code: OnceCell::new(),
            values: BufferMut::empty_aligned_in(Alignment::of::<V>(), allocator.clone()),
            values_nulls: BitBufferMut::empty_in(allocator.clone()),
            max_dict_len,
            allocator,
        }
    }

    /// Returns whether the dictionary holds no values.
    pub(super) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the codes of the longest prefix of `values` that fits the dictionary constraints,
    /// adding the values it has not seen yet. `mask` holds the validity of `values`, and `key`
    /// gives the lookup key of a value.
    pub(super) fn encode(
        &mut self,
        values: &[V],
        mask: &Mask,
        key: impl Fn(V) -> K,
    ) -> BufferMut<Code> {
        let mut codes = BufferMut::<Code>::with_capacity_in(values.len(), self.allocator.clone());
        match mask {
            Mask::AllTrue(_) => {
                for &value in values {
                    let Some(code) = self.encode_value(key(value), value) else {
                        break;
                    };
                    // SAFETY: `codes` holds `values.len()` codes, at most one per value.
                    unsafe { codes.push_unchecked(code) }
                }
            }
            Mask::AllFalse(_) => {
                if let Some(code) = self.encode_null() {
                    // SAFETY: `codes` holds `values.len()` codes.
                    unsafe { codes.push_n_unchecked(code, values.len()) }
                }
            }
            Mask::Values(v) => {
                for (&value, valid) in values.iter().zip(v.bit_buffer()) {
                    let code = if valid {
                        self.encode_value(key(value), value)
                    } else {
                        self.encode_null()
                    };
                    let Some(code) = code else {
                        break;
                    };
                    // SAFETY: `codes` holds `values.len()` codes, at most one per value.
                    unsafe { codes.push_unchecked(code) }
                }
            }
        }
        codes
    }

    /// Returns the values and their validity bits, and clears the dictionary for a new round of
    /// encoding.
    pub(super) fn reset(&mut self) -> (Buffer<V>, BitBuffer) {
        self.lookup.clear();
        self.null_code = OnceCell::new();
        let values = mem::replace(
            &mut self.values,
            BufferMut::empty_aligned_in(Alignment::of::<V>(), self.allocator.clone()),
        );
        let nulls = mem::replace(
            &mut self.values_nulls,
            BitBufferMut::empty_in(self.allocator.clone()),
        );
        (values.freeze(), nulls.freeze())
    }

    /// Returns `None` when assigning a code would exceed the dictionary constraints,
    /// and callers should stop encoding after the current prefix.
    fn encode_value(&mut self, key: K, value: V) -> Option<Code> {
        match self.lookup.entry(key) {
            Entry::Occupied(o) => Some(*o.get()),
            Entry::Vacant(vac) => {
                if is_full(&self.values, self.max_dict_len) {
                    return None;
                }
                let next_code = next_code(&self.values);
                self.values.push(value);
                self.values_nulls.append_true();
                Some(*vac.insert(next_code))
            }
        }
    }

    /// Returns `None` when assigning the null code would exceed the dictionary constraints,
    /// and callers should stop encoding after the current prefix.
    fn encode_null(&mut self) -> Option<Code> {
        if let Some(code) = self.null_code.get() {
            return Some(*code);
        }

        if is_full(&self.values, self.max_dict_len) {
            return None;
        }

        let code = next_code(&self.values);
        self.values.push(V::default());
        self.values_nulls.append_false();
        self.null_code
            .set(code)
            .ok()
            .vortex_expect("null code is initialized once");
        Some(code)
    }
}

/// Returns whether a dictionary of `values` is full, so a new entry exceeds the constraints. The
/// first entry is always admitted, even over the budget, so that the caller makes progress.
fn is_full<V>(values: &BufferMut<V>, max_dict_len: usize) -> bool {
    !values.is_empty() && values.len() >= max_dict_len
}

/// Returns the code of the next entry of a dictionary of `values`.
fn next_code<V, Code: UnsignedPType>(values: &BufferMut<V>) -> Code {
    Code::from_usize(values.len())
        .unwrap_or_else(|| vortex_panic!("{} has to fit into {}", values.len(), Code::PTYPE))
}
