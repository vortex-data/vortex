// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::display::ExprDisplay;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// A predicate that a switch turns on and off: `on ⇒ predicate`.
///
/// While on, the predicate is evaluated for the demanded rows. While off, the expression is `true`
/// and the predicate is not evaluated, as its demand is empty. An engine can push predicates that
/// only remove rows it removes again, such as the dynamic filters of a hash join or a top-N
/// operator, and a scan turns such a predicate off when evaluating it costs more than it saves.
///
/// Applying the predicate is always permitted, so its stats proofs are those of the predicate.
#[derive(Clone)]
pub struct Switch;

/// The state of a [`Switch`], shared by every copy of its expression.
#[derive(Clone, Debug)]
pub struct SwitchState(Arc<SwitchInner>);

#[derive(Debug)]
struct SwitchInner {
    on: AtomicBool,
    /// The largest percentage of its input rows that the predicate can keep and still be worth
    /// its evaluation.
    max_kept_percent: u8,
}

impl SwitchState {
    /// A new switch, on, over a predicate that is worth its evaluation while it keeps at most
    /// `max_kept_percent` percent of its input rows.
    pub fn new(max_kept_percent: u8) -> Self {
        Self(Arc::new(SwitchInner {
            on: AtomicBool::new(true),
            max_kept_percent: max_kept_percent.min(100),
        }))
    }

    /// Whether the predicate is applied.
    pub fn is_on(&self) -> bool {
        self.0.on.load(Ordering::Relaxed)
    }

    /// Turn the predicate on or off.
    pub fn set(&self, on: bool) {
        self.0.on.store(on, Ordering::Relaxed);
    }

    /// The largest fraction of its input rows that the predicate can keep and still be worth its
    /// evaluation.
    pub fn max_selectivity(&self) -> f64 {
        f64::from(self.0.max_kept_percent) / 100.0
    }
}

impl Display for SwitchState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.is_on() { "on" } else { "off" })
    }
}

/// Equality and hashing use the identity of the state, so turning a switch does not change the
/// identity of its expression.
impl PartialEq for SwitchState {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SwitchState {}

impl Hash for SwitchState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl ScalarFnVTable for Switch {
    type Options = SwitchState;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.switch");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("predicate"),
            _ => unreachable!("Invalid child index {child_idx} for Switch expression"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "switch({})", expr.display_child(0))
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(arg_dtypes[0], DType::Bool(_)),
            "Switch requires a boolean predicate, got {}",
            arg_dtypes[0]
        );
        Ok(arg_dtypes[0].clone())
    }

    fn execute(
        &self,
        state: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let predicate = args.get(0)?;
        if !state.is_on() {
            let on = Scalar::bool(true, predicate.dtype().nullability());
            return Ok(ConstantArray::new(on, args.row_count()).into_array());
        }
        Ok(predicate)
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::switch;
    use crate::scalar_fn::ScalarFnVTableExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn applies_the_predicate_while_on() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let state = SwitchState::new(50);
        let expr = Switch.new_expr(state.clone(), [lt(col("a"), lit(5i32))]);
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();

        assert_arrays_eq!(
            input.clone().apply(&expr)?,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );

        state.set(false);
        assert_arrays_eq!(
            input.apply(&expr)?,
            BoolArray::from_iter([true, true, true]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn new_switches_are_on() {
        let expr = switch(lit(false), 150);
        assert!(expr.as_::<Switch>().is_on());
        assert_eq!(expr.as_::<Switch>().max_selectivity(), 1.0);
    }
}
