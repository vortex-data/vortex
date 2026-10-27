// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Emitting one batch.

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::Step;

/// Emits one batch it was built with.
pub(crate) struct OnceSource(Option<ArrayRef>);

impl OnceSource {
    pub(crate) fn new(batch: ArrayRef) -> Self {
        Self(Some(batch))
    }
}

impl Source for OnceSource {}

impl Operator for OnceSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match self.0.take() {
            Some(batch) => Step::Final(batch),
            None => Step::End,
        })
    }
}
