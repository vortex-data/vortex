// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::Operator;
use crate::Canonical;
use std::sync::Arc;
use vortex_error::VortexResult;

/// An executor that runs an operator tree.
///
/// The executor performs common subtree elimination by creating BatchExecution nodes that hold
/// shared futures to the underlying execution.
///
/// It also finds sub-graphs of pipeline operators and executes them as a [`Pipeline`]
pub struct Executor {}

impl Executor {
    pub async fn execute(&mut self, operator: Arc<dyn Operator>) -> VortexResult<Canonical> {
        todo!()
    }
}
