// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::check_patch_indices;

impl BoolArray {
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let len = self.len();
        let offset = patches.offset();
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<BoolArray>(ctx)?;

        let patched_validity =
            self.validity()?
                .patch(len, offset, patches.indices(), &values.validity()?, ctx)?;

        let bit_buffer = self.into_bit_buffer();
        let mut own_values = bit_buffer
            .try_into_mut()
            .unwrap_or_else(|bb| BitBufferMut::copy_from(&bb));
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            let indices = indices.as_slice::<I>();
            // Checked up front so `set_to` below cannot exceed the buffer; see
            // `check_patch_indices` for why construction is not enough.
            check_patch_indices(indices, offset, len)?;
            for (idx, value) in indices.iter().zip_eq(values.bit_buffer_view().iter()) {
                #[allow(clippy::cast_possible_truncation)]
                own_values.set_to(*idx as usize - offset, value);
            }
        });

        Ok(Self::new(own_values.freeze(), patched_validity))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;

    /// The reported crash: `BitBufferMut::set` panicked with
    /// "index 402653634 exceeds len 1024" while patching bool validity read from a
    /// file. The indices are unsorted so the last-element maximum check in
    /// `Patches::new` does not see the out-of-range value, and sortedness is only
    /// asserted under `debug_assertions`.
    #[test]
    fn patch_rejects_out_of_range_index() {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::from(BitBuffer::new_set(8));
        let patches = unsafe {
            Patches::new_unchecked(
                8,
                0,
                buffer![1u64, 402_653_634, 2].into_array(),
                BoolArray::from_iter([false, false, false]).into_array(),
                None,
                None,
            )
        };

        let err = array
            .patch(&patches, &mut ctx)
            .expect_err("out-of-range patch index must be rejected");
        assert!(
            err.to_string().contains("402653634"),
            "unexpected error: {err}"
        );
    }

    /// A valid patch set must still be applied — the check must not over-reject.
    #[test]
    fn patch_accepts_in_range_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::from(BitBuffer::new_set(4));
        let patches = unsafe {
            Patches::new_unchecked(
                4,
                0,
                buffer![1u64, 3].into_array(),
                BoolArray::from_iter([false, false]).into_array(),
                None,
                None,
            )
        };

        let patched = array.patch(&patches, &mut ctx).unwrap();
        let expected = BoolArray::from_iter([true, false, true, false]);
        assert_arrays_eq!(patched, expected, &mut ctx);
    }

    #[test]
    fn patch_sliced_bools() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = BoolArray::from(BitBuffer::new_set(12));
        let sliced = arr.into_array().slice(4..12).unwrap();
        let expected = BoolArray::from_iter([true; 8]);
        assert_arrays_eq!(sliced, expected, &mut ctx);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = BoolArray::from(BitBuffer::new_set(15));
        let sliced = arr.into_array().slice(4..15).unwrap();
        let expected = BoolArray::from_iter([true; 11]);
        assert_arrays_eq!(sliced, expected, &mut ctx);
    }
}
