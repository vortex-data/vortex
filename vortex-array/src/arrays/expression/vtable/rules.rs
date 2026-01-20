// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExpressionArray;
use crate::arrays::ExpressionVTable;
use crate::arrays::StructArray;
use crate::expr::Pack;
use crate::expr::Root;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ExpressionVTable> =
    ReduceRuleSet::new(&[&ExpressionRootRule, &ExpressionPackToStructRule]);

/// A root expression reduces to just the scope array.
#[derive(Debug)]
struct ExpressionRootRule;
impl ArrayReduceRule<ExpressionVTable> for ExpressionRootRule {
    fn reduce(&self, array: &ExpressionArray) -> VortexResult<Option<ArrayRef>> {
        if array.expression().is::<Root>() {
            Ok(Some(array.input.clone()))
        } else {
            Ok(None)
        }
    }
}

/// Convert pack expressions into StructArrays.
#[derive(Debug)]
struct ExpressionPackToStructRule;
impl ArrayReduceRule<ExpressionVTable> for ExpressionPackToStructRule {
    fn reduce(&self, array: &ExpressionArray) -> VortexResult<Option<ArrayRef>> {
        let Some(pack) = array.expression().as_opt::<Pack>() else {
            return Ok(None);
        };

        // Transform each child expression into an array.
        let fields = array
            .expression()
            .children()
            .iter()
            .map(|expr| {
                ExpressionArray::try_new(expr.clone(), array.input.clone())
                    .map(IntoArray::into_array)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::try_new(
                pack.names.clone(),
                fields,
                array.len(),
                Validity::from(pack.nullability),
            )?
            .into_array(),
        ))
    }
}
