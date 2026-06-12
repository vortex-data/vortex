// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::BoundCall;
use crate::expr::placeholder::PlaceholderRef;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Internal lazy-array marker for unresolved placeholders.
#[derive(Clone)]
pub struct PlaceholderFn;

impl ScalarFnVTable for PlaceholderFn {
    type Options = PlaceholderRef;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.placeholder");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        unreachable!(
            "PlaceholderFn expression does not have children, got index {}",
            child_idx
        )
    }

    fn fmt_sql(
        &self,
        placeholder: &Self::Options,
        _call: &BoundCall,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}()", placeholder.display_name())
    }

    fn return_dtype(&self, placeholder: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(placeholder.dtype().clone())
    }

    fn execute(
        &self,
        placeholder: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("unresolved placeholder {}", placeholder.id())
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}
