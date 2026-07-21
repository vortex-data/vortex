// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Delta;
use crate::TransposedBool;
use crate::delta::array::DeltaArrayExt;

impl ValidityVTable<Delta> for Delta {
    fn validity(array: ArrayView<'_, Delta>) -> VortexResult<Validity> {
        let start = array.offset();
        let stop = start + array.len();
        let validity = match array.deltas().validity()? {
            Validity::Array(mask) => {
                let Some(mask) = mask.as_opt::<Bool>() else {
                    vortex_bail!(
                        "DeltaArray storage validity must be a BoolArray, got {}",
                        mask.encoding_id()
                    );
                };
                Validity::Array(TransposedBool::try_new(mask.to_bit_buffer())?.into_array())
            }
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
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::*;
    use crate::TransposedBool;

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
}
