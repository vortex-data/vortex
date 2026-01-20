// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::expr::Expression;
use vortex_array::expr::Literal;
use vortex_array::expr::Root;
use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::readers::constant::ConstantReader;
use crate::v2::readers::scalar_fn::ScalarFnReader;

impl dyn Reader + '_ {
    /// Apply the expression to this reader, producing a new reader in constant time.
    ///
    /// FIXME(ngates): how should we differentiate between prune, filter, and project expressions?
    pub fn apply(self: Arc<Self>, expr: &Expression) -> VortexResult<ReaderRef> {
        // If the expression is a root, return self.
        if expr.is::<Root>() {
            return Ok(self);
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(Arc::new(ConstantReader::new(
                scalar.clone(),
                self.row_count(),
            )));
        }

        let row_count = self.row_count();

        // Otherwise, collect the child readers.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let reader: ReaderRef = Arc::new(ScalarFnReader::try_new(
            expr.scalar_fn().clone(),
            children,
            row_count,
        )?);

        // Optimize the resulting reader.
        reader.optimize()
    }
}
