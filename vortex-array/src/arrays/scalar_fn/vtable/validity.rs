// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::vtable::ScalarFn;
use crate::arrays::scalar_fn::vtable::scalar_fn_array_expr;
use crate::validity::Validity;

impl ValidityVTable<ScalarFn> for ScalarFn {
    fn validity(array: ArrayView<'_, ScalarFn>) -> VortexResult<Validity> {
        let expr = scalar_fn_array_expr(array)?.validity()?;
        let input = ConstantArray::new(true, array.len()).into_array();
        Ok(Validity::Array(input.apply(&expr)?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::arrays::scalar_fn::vtable::ScalarFnFactoryExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    #[test]
    fn scalar_fn_validity_stays_lazy() -> VortexResult<()> {
        let lhs = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
        let rhs = BoolArray::from_iter([Some(true), Some(false), None]).into_array();
        let predicate = Binary.try_new_array(lhs.len(), Operator::And, [lhs, rhs])?;

        let Validity::Array(validity_array) = predicate.validity()? else {
            panic!("scalar function validity should be represented as an array");
        };

        let validity_scalar_fn = validity_array
            .as_opt::<ScalarFn>()
            .expect("validity should remain a lazy scalar function array");
        assert!(validity_scalar_fn.scalar_fn().is::<Binary>());

        let validity_mask =
            validity_array.execute::<Mask>(&mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(validity_mask.all_true());

        Ok(())
    }
}
