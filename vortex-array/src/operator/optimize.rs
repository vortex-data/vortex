// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::operator::{
    Operator,
    OperatorRef,
};

impl dyn Operator + '_ {
    /// Optimize the operator tree rooted at this operator by applying local
    /// optimizations such as reducing redundant operators.
    pub fn optimize(self: Arc<Self>) -> VortexResult<OperatorRef> {
        let children = self
            .children()
            .iter()
            .map(|child| child.clone().optimize())
            .try_collect()?;

        let mut operator = self.with_children(children)?;
        operator = operator.reduce_children()?.unwrap_or(operator);

        let parent = operator.clone();
        for (idx, child) in operator.children().iter().enumerate() {
            if let Some(new_operator) = child.reduce_parent(parent.clone(), idx)? {
                return Ok(new_operator);
            }
        }

        Ok(operator)
    }
}
