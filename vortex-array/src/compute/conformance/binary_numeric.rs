use num_traits::Num;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};
use vortex_scalar::{NumericOperator, PrimitiveScalar, Scalar};

use crate::arrays::ConstantArray;
use crate::compute::{numeric, scalar_at};
use crate::{Array, ArrayRef, ToCanonical};

fn to_vec_of_scalar(array: &dyn Array) -> Vec<Scalar> {
    // Not fast, but obviously correct
    (0..array.len())
        .map(|index| scalar_at(array, index))
        .collect::<VortexResult<Vec<_>>>()
        .vortex_unwrap()
}

pub fn test_numeric<T: NativePType + Num + Copy>(array: ArrayRef)
where
    Scalar: From<T>,
{
    let canonicalized_array = array.to_primitive().vortex_unwrap();
    let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

    let one = T::from(1)
        .ok_or_else(|| vortex_err!("could not convert 1 into array native type"))
        .vortex_unwrap();
    let scalar_one = Scalar::from(one).cast(array.dtype()).vortex_unwrap();

    let operators: [NumericOperator; 6] = [
        NumericOperator::Add,
        NumericOperator::Sub,
        NumericOperator::RSub,
        NumericOperator::Mul,
        NumericOperator::Div,
        NumericOperator::RDiv,
    ];

    for operator in operators {
        assert_eq!(
            to_vec_of_scalar(
                &numeric(
                    &array,
                    &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
                    operator
                )
                .vortex_unwrap()
            ),
            original_values
                .iter()
                .map(|x| x
                    .as_primitive()
                    .checked_binary_numeric(&scalar_one.as_primitive(), operator)
                    .vortex_expect("numeric operator overflow"))
                .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
                .collect::<Vec<Scalar>>(),
            "({:?}) {} (Constant array of {}) did not produce expected results",
            array,
            operator,
            scalar_one,
        );

        assert_eq!(
            to_vec_of_scalar(
                &numeric(
                    &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
                    &array,
                    operator
                )
                .vortex_unwrap()
            ),
            original_values
                .iter()
                .map(|x| scalar_one
                    .as_primitive()
                    .checked_binary_numeric(&x.as_primitive(), operator)
                    .vortex_expect("numeric operator overflow"))
                .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
                .collect::<Vec<_>>(),
            "(Constant array of {}) {} ({:?}) did not produce expected results",
            scalar_one,
            operator,
            array,
        );
    }
}
