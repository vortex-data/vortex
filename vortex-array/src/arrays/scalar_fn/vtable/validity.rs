// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::NullArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ArrayExpr;
use crate::arrays::scalar_fn::vtable::FakeEq;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::expr::Expression;
use crate::expr::ScalarFn;
use crate::expr::lit;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn validity(array: &ScalarFnArray) -> VortexResult<Validity> {
        let inputs: Arc<[_]> = array
            .children
            .iter()
            .map(|child| {
                if let Some(scalar) = child.as_constant() {
                    return Ok(lit(scalar));
                }
                Expression::try_new(ScalarFn::new(ArrayExpr, FakeEq(child.clone())), [])
            })
            .collect::<VortexResult<_>>()?;

        let expr = Expression::try_new(array.scalar_fn.clone(), inputs)?;
        let validity_expr = array.scalar_fn().validity(&expr)?;

        // We can evaluate the validity expression against an empty scope because we know all
        // leaves are ArrayExpr.
        Ok(Validity::Array(
            validity_expr.evaluate(&NullArray::new(array.len()).into_array())?,
        ))
    }

    fn validity_mask(array: &ScalarFnArray) -> Mask {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let output = array
            .to_array()
            .execute::<CanonicalOutput>(&mut ctx)
            .vortex_expect("Validity mask computation should be fallible");
        match output {
            CanonicalOutput::Constant(c) => Mask::new(array.len, c.scalar().is_valid()),
            CanonicalOutput::Array(a) => a
                .into_array()
                .validity()
                .vortex_expect("cannot fail")
                .to_mask(array.len()),
        }
    }
}
