// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::Delta;
use crate::TransposedBool;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;

impl ValidityVTable<Delta> for Delta {
    fn validity(array: ArrayView<'_, Delta>) -> VortexResult<Validity> {
        let start = array.offset();
        let stop = start + array.len();
        let validity = match array.deltas().validity()? {
            Validity::Array(mask) => Validity::Array(TransposedBool::try_new(mask)?.into_array()),
            validity => validity,
        };
        validity.slice(start..stop)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::*;
    use crate::TransposedBool;
    use crate::delta::array::delta_compress::delta_compress;

    #[test]
    fn validity_is_lazy_for_cross_chunk_slice() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0u32..2048).map(|value| (value % 3 != 0).then_some(value)),
        );
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        let sliced = delta.slice(1000..1050)?;

        let Validity::Array(validity) = sliced.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert!(validity.is::<TransposedBool>());
        assert_arrays_eq!(
            validity,
            BoolArray::from_iter((1000u32..1050).map(|value| value % 3 != 0)),
            &mut ctx
        );
        Ok(())
    }

    /// Regression: the deltas' storage validity is not always a raw `Bool` array — slicing or a
    /// file round-trip can leave it wrapped in a lazy encoding such as `vortex.slice`. The Delta
    /// validity must accept it rather than bail.
    #[test]
    fn validity_handles_slice_encoded_storage_validity() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0u32..2048).map(|value| (value % 3 != 0).then_some(value)),
        );
        let (bases, deltas) = delta_compress(&primitive, &mut ctx)?;

        // Rebuild the deltas with a lazily slice-encoded validity, as produced when the deltas
        // child is sliced and the validity encoding has no static slice reduction.
        let Validity::Array(storage_validity) = deltas.validity()? else {
            vortex_bail!("expected array-backed storage validity")
        };
        let lazy_validity = SliceArray::try_new(storage_validity, 0..deltas.len())?.into_array();
        let deltas = PrimitiveArray::new(deltas.to_buffer::<u32>(), Validity::Array(lazy_validity));
        let delta = Delta::try_new(bases.into_array(), deltas.into_array(), 0, primitive.len())?;

        let Validity::Array(validity) = delta.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(
            validity,
            BoolArray::from_iter((0u32..2048).map(|value| value % 3 != 0)),
            &mut ctx
        );
        assert_arrays_eq!(delta, primitive, &mut ctx);
        Ok(())
    }
}
