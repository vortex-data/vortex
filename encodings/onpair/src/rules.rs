// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

pub(crate) static RULES: ParentRuleSet<OnPair> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&EmptyComparisonRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(OnPair)),
    ParentRuleSet::lift(&CastReduceAdaptor(OnPair)),
]);

/// Empty-string comparisons depend only on lengths and validity, without reading string codes.
#[derive(Debug)]
struct EmptyComparisonRule;

impl ArrayParentReduceRule<OnPair> for EmptyComparisonRule {
    type Parent = ExactScalarFn<Binary>;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, OnPair>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(operator) = parent.options.to_compare_operator() else {
            return Ok(None);
        };
        let scalar_fn = parent.as_::<ScalarFn>();
        let (operator, other) = match child_idx {
            0 => (operator, scalar_fn.get_child(1)),
            1 => (operator.swap(), scalar_fn.get_child(0)),
            _ => return Ok(None),
        };
        let Some(constant) = other.as_constant() else {
            return Ok(None);
        };
        let empty = match constant.dtype() {
            DType::Utf8(_) => constant.as_utf8().is_empty(),
            DType::Binary(_) => constant.as_binary().is_empty(),
            _ => return Ok(None),
        };
        if empty != Some(true) {
            return Ok(None);
        }

        let result = match operator {
            CompareOperator::Gte => ConstantArray::new(true, child.len()).into_array(),
            CompareOperator::Lt => ConstantArray::new(false, child.len()).into_array(),
            _ => {
                let lengths = child.uncompressed_lengths();
                lengths.binary(
                    ConstantArray::new(Scalar::zero_value(lengths.dtype()), lengths.len())
                        .into_array(),
                    operator.into(),
                )?
            }
        }
        .cast(parent.dtype().clone())?;
        Ok(Some(match child.validity()? {
            Validity::NonNullable | Validity::AllValid => result,
            Validity::Array(validity) => result.mask(validity)?,
            Validity::AllInvalid => {
                ConstantArray::new(Scalar::null(parent.dtype().clone()), child.len()).into_array()
            }
        }))
    }
}
