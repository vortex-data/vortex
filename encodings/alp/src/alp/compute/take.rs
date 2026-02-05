// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::TakeReduce;
use vortex_array::arrays::TakeReduceAdaptor;
use vortex_array::compute::take;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPVTable;

fn take_alp(array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
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

impl TakeReduce for ALPVTable {
    fn take(array: &ALPArray, indices: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        take_alp(array, indices).map(Some)
    }
}

impl ALPVTable {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_buffer::buffer;

    use crate::alp_encode;

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    fn test_take_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let alp = alp_encode(&array.to_primitive(), None).unwrap();
        test_take_conformance(alp.as_ref());
    }
}
