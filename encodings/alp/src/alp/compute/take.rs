// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{ALPArray, ALPVTable};

impl TakeKernel for ALPVTable {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|patches| {
                patches.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::new(taken_encoded, array.exponents(), taken_patches).into_array())
    }
}

register_kernel!(TakeKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::vtable::ArrayVTableExt;
    use vortex_buffer::buffer;

    use crate::ALPVTable;

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    fn test_take_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let alp = ALPVTable
            .as_vtable()
            .encode(&array.to_canonical(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(alp.as_ref());
    }
}
