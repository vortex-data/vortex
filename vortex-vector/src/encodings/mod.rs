// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bitpacked;
// mod compare;
// pub mod primitive;
// pub mod validity;

use crate::pipeline::Pipeline;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub trait Encoding {
    /// [`DType`] and length of the node are passed down in the bind context.
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Pipeline>>;
}

/// Context required for binding a node.
///
/// During the bind phase, context is passed down from parent nodes to child nodes.
pub struct BindContext<'a> {
    pub len: usize,
    pub dtype: &'a DType,
    pub stats: Option<&'a StatsSet>,
}
