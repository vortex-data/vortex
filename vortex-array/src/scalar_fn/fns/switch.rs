// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
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

/// The input rows for which a scan first skips a predicate that keeps too many rows. Each further
/// skip in a row doubles it, up to [`MAX_PAUSE_DOUBLINGS`] times, as DuckDB increases the pause of
/// its optional filters.
pub const PAUSE_ROWS: u64 = 1 << 20;

/// The most times the pause of a predicate doubles.
const MAX_PAUSE_DOUBLINGS: u32 = 4;

/// The state of a [`Switch`], shared by every copy of its expression.
///
/// Every scan that evaluates the predicate, such as the scans of the files of a table, shares the
/// measurements of its selectivity and the decision to skip it. A predicate that was paused is
/// measured again by one evaluation at a time while the others skip it, so that scans running side
/// by side do not all pay again for a predicate that keeps most rows.
#[derive(Clone, Debug)]
pub struct SwitchState(Arc<SwitchInner>);

#[derive(Debug)]
struct SwitchInner {
    on: AtomicBool,
    /// The largest percentage of its input rows that the predicate can keep and still be worth
    /// its evaluation.
    max_kept_percent: u8,
    pause: Mutex<Pause>,
}

#[derive(Debug, Default)]
struct Pause {
    /// Whether an evaluation since the predicate was turned on has reported its selectivity.
    measured: bool,
    /// Whether an evaluation is measuring the predicate.
    measuring: bool,
    /// Input rows still to skip while off.
    skip_rows: u64,
    /// The number of pauses in a row, which sets the length of the next pause.
    pauses: u32,
    /// The version of the predicate when it was turned off.
    paused_version: Option<u64>,
}

impl SwitchState {
    /// A new switch, on, over a predicate that is worth its evaluation while it keeps at most
    /// `max_kept_percent` percent of its input rows.
    pub fn new(max_kept_percent: u8) -> Self {
        Self(Arc::new(SwitchInner {
            on: AtomicBool::new(true),
            max_kept_percent: max_kept_percent.min(100),
            pause: Mutex::default(),
        }))
    }

    /// Whether the predicate is applied.
    pub fn is_on(&self) -> bool {
        self.0.on.load(Ordering::Relaxed)
    }

    /// The largest fraction of its input rows that the predicate can keep and still be worth its
    /// evaluation.
    pub fn max_selectivity(&self) -> f64 {
        f64::from(self.0.max_kept_percent) / 100.0
    }

    /// Whether a scan should evaluate the predicate over its next `input_rows` rows, for a
    /// predicate at `version`, such as the version of a dynamic expression.
    ///
    /// While off, the skipped rows count towards the pause, and a new version ends it. While
    /// another evaluation measures the predicate, it is skipped. A scan that evaluates the
    /// predicate first calls [`Self::begin_evaluation`].
    pub fn should_evaluate(&self, version: Option<u64>, input_rows: u64) -> bool {
        let mut pause = self.0.pause.lock();
        if !self.is_on() {
            if version != pause.paused_version {
                pause.pauses = 0;
            } else {
                pause.skip_rows = pause.skip_rows.saturating_sub(input_rows);
                if pause.skip_rows > 0 {
                    return false;
                }
            }
            pause.skip_rows = 0;
            pause.measured = false;
            self.0.on.store(true, Ordering::Relaxed);
        }
        // A predicate that was paused is measured by one evaluation at a time, while the others
        // keep skipping it. Before its first pause, every evaluation measures it: skipping a
        // selective predicate on the first chunks of a scan can cost more than the predicate.
        pause.measured || !pause.measuring || pause.pauses == 0
    }

    /// Start an evaluation of the predicate, which then reports with [`Self::report`] or gives
    /// up with [`Self::abandon`]. Returns `false` if another evaluation measures the predicate,
    /// in which case the scan skips it.
    pub fn begin_evaluation(&self) -> bool {
        let mut pause = self.0.pause.lock();
        if pause.measured {
            return true;
        }
        if pause.measuring {
            return pause.pauses == 0;
        }
        pause.measuring = true;
        true
    }

    /// Give up an evaluation started with [`Self::begin_evaluation`] without a measurement.
    pub fn abandon(&self) {
        self.0.pause.lock().measuring = false;
    }

    /// Report that an evaluation of the predicate at `version` kept `output_rows` of
    /// `input_rows` rows. A predicate that keeps more than it may is turned off for a pause.
    ///
    /// Only a `standalone` evaluation, one over rows that no other predicate filtered, measures
    /// the selectivity of the predicate itself. A predicate evaluated after others only sees the
    /// rows they keep, of which it can keep most even when it is selective on its own, and it
    /// costs little there, so such an evaluation never turns the switch off.
    pub fn report(
        &self,
        version: Option<u64>,
        input_rows: u64,
        output_rows: u64,
        standalone: bool,
    ) {
        let mut pause = self.0.pause.lock();
        pause.measuring = false;
        if input_rows == 0 {
            return;
        }
        pause.measured = true;
        if !standalone {
            return;
        }
        #[expect(clippy::cast_precision_loss, reason = "a selectivity estimate")]
        let selectivity = output_rows as f64 / input_rows as f64;
        if selectivity <= self.max_selectivity() {
            pause.pauses = 0;
            return;
        }
        // Evaluations that started before the pause do not extend it.
        if !self.is_on() {
            return;
        }
        pause.skip_rows = PAUSE_ROWS << pause.pauses.min(MAX_PAUSE_DOUBLINGS);
        pause.pauses += 1;
        pause.paused_version = version;
        self.0.on.store(false, Ordering::Relaxed);
        tracing::debug!(
            selectivity,
            skip_rows = pause.skip_rows,
            "pausing switched predicate"
        );
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

        // A measurement that keeps every row turns the switch off.
        assert!(state.begin_evaluation());
        state.report(None, 3, 3, true);
        assert!(!state.is_on());
        assert_arrays_eq!(
            input.apply(&expr)?,
            BoolArray::from_iter([true, true, true]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn evaluations_side_by_side_measure_cheap_predicates() {
        let state = SwitchState::new(50);
        assert!(state.begin_evaluation());
        assert!(state.should_evaluate(None, 100));
        assert!(state.begin_evaluation());
    }

    #[test]
    fn pauses_predicates_that_keep_too_many_rows() -> VortexResult<()> {
        let pause = usize::try_from(PAUSE_ROWS)?;
        let state = SwitchState::new(50);
        assert!(state.begin_evaluation());
        state.report(None, 100, 90, true);
        assert!(!state.is_on());

        // The skipped rows count towards the pause, after which one evaluation measures again.
        assert!(!state.should_evaluate(None, pause as u64 - 1));
        assert!(state.should_evaluate(None, 1));
        assert!(state.is_on());
        assert!(state.begin_evaluation());
        assert!(!state.should_evaluate(None, 100));
        assert!(!state.begin_evaluation());

        // An abandoned measurement lets another evaluation measure.
        state.abandon();
        assert!(state.begin_evaluation());

        // A second pause in a row is twice as long.
        state.report(None, 100, 90, true);
        assert!(!state.should_evaluate(None, PAUSE_ROWS));
        assert!(!state.should_evaluate(None, PAUSE_ROWS - 1));
        assert!(state.should_evaluate(None, 1));
        Ok(())
    }

    #[test]
    fn evaluations_after_other_predicates_do_not_pause() {
        let state = SwitchState::new(50);
        assert!(state.begin_evaluation());
        state.report(None, 100, 90, false);
        assert!(state.is_on());
        assert!(state.should_evaluate(None, 100));
    }

    #[test]
    fn a_new_version_ends_the_pause() {
        let state = SwitchState::new(50);
        assert!(state.begin_evaluation());
        state.report(Some(1), 100, 90, true);
        assert!(!state.should_evaluate(Some(1), 100));
        assert!(state.should_evaluate(Some(2), 100));
    }

    #[test]
    fn new_switches_are_on() {
        let expr = switch(lit(false), 150);
        assert!(expr.as_::<Switch>().is_on());
        assert_eq!(expr.as_::<Switch>().max_selectivity(), 1.0);
    }
}
