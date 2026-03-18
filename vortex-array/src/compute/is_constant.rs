// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

/// Computes whether an array has constant values.
///
/// **Deprecated**: Use [`crate::aggregate_fn::fns::is_constant::is_constant`] instead.
#[deprecated(note = "Use crate::aggregate_fn::fns::is_constant::is_constant instead")]
pub fn is_constant(array: &ArrayRef) -> VortexResult<Option<bool>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    Ok(Some(crate::aggregate_fn::fns::is_constant::is_constant(
        array, &mut ctx,
    )?))
}

/// Computes whether an array has constant values.
///
/// **Deprecated**: Use [`crate::aggregate_fn::fns::is_constant::is_constant`] instead.
#[deprecated(note = "Use crate::aggregate_fn::fns::is_constant::is_constant instead")]
pub fn is_constant_opts(array: &ArrayRef, _opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    Ok(Some(crate::aggregate_fn::fns::is_constant::is_constant(
        array, &mut ctx,
    )?))
}

/// When calling `is_constant` the children are all checked for constantness.
/// This enum decide at each precision/cost level the constant check should run as.
/// The cost increase as we move down the list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cost {
    /// Only apply constant time computation to estimate constantness.
    Negligible,
    /// Allow the encoding to do a linear amount of work to determine is constant.
    Specialized,
    /// Same as linear, but when necessary canonicalize the array and check is constant.
    Canonicalize,
}

/// Configuration for [`is_constant_opts`] operations.
#[derive(Clone, Debug)]
pub struct IsConstantOpts {
    /// What precision cost trade off should be used
    pub cost: Cost,
}

impl Default for IsConstantOpts {
    fn default() -> Self {
        Self {
            cost: Cost::Canonicalize,
        }
    }
}

impl super::Options for IsConstantOpts {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl IsConstantOpts {
    pub fn is_negligible_cost(&self) -> bool {
        self.cost == Cost::Negligible
    }
}
