// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

mod binary;
mod cast;
mod conditional;
mod nulls;
mod structural;

pub(crate) use binary::BinaryBoolean;
pub(crate) use binary::BinaryNullComparison;
pub(crate) use cast::CastLiteralOrIdentity;
pub(crate) use conditional::ConstantMask;
pub(crate) use conditional::ConstantZip;
pub(crate) use nulls::CaseWhenToFillNull;
pub(crate) use nulls::RemoveRedundantFillNull;
pub(crate) use structural::GetItemFromPack;
pub(crate) use structural::MergeToPack;
pub(crate) use structural::SelectFromPack;

/// Shared reference to a rewrite rule.
pub type OptimizerRuleRef = Arc<dyn OptimizerRule>;

/// An equivalence rewrite for bound expressions with a particular root node implementation.
///
/// The optimizer invokes a rule only when the expression's root ID equals
/// [`Self::expression_id`]. Returning `None` means the rule does not match. A replacement must be
/// semantically equivalent to the input and have exactly the same dtype, including nullability.
/// The optimizer verifies the dtype and rejects unchanged replacements.
pub trait OptimizerRule: Debug + Send + Sync + 'static {
    /// Returns a diagnostic name for this rule.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Returns the expression node ID handled by this rule.
    fn expression_id(&self) -> ExpressionId;

    /// Try to rewrite `expr` to a semantically equivalent bound expression.
    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>>;
}

/// Rewrite rules grouped by the expression node ID they handle.
#[derive(Debug)]
pub struct OptimizerRuleRegistry {
    rules: HashMap<ExpressionId, Vec<OptimizerRuleRef>>,
}

impl OptimizerRuleRegistry {
    /// Create an empty rule registry.
    pub fn empty() -> Self {
        Self {
            rules: HashMap::default(),
        }
    }

    /// Register a rule after existing rules for the same expression node ID.
    pub fn register<R: OptimizerRule>(&mut self, rule: R) {
        self.rules
            .entry(rule.expression_id())
            .or_default()
            .push(Arc::new(rule));
    }

    /// Return rules for `expression_id` in registration order.
    pub(super) fn get(&self, expression_id: ExpressionId) -> Option<&[OptimizerRuleRef]> {
        self.rules.get(&expression_id).map(Vec::as_slice)
    }
}

impl Default for OptimizerRuleRegistry {
    fn default() -> Self {
        let mut registry = Self::empty();

        registry.register(BinaryBoolean);
        registry.register(BinaryNullComparison);
        registry.register(CastLiteralOrIdentity);
        registry.register(GetItemFromPack);
        registry.register(MergeToPack);
        registry.register(SelectFromPack);
        registry.register(RemoveRedundantFillNull);
        registry.register(CaseWhenToFillNull);
        registry.register(ConstantMask);
        registry.register(ConstantZip);

        registry
    }
}

fn preserve_dtype(replacement: BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    if replacement.dtype() == dtype {
        return Ok(replacement);
    }
    Cast.try_new_bound_expr(dtype.clone(), [replacement])
}
