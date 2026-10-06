// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Handoff between stages: the `Next` constructor.

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::planning::planner::Planner;

/// A reusable constructor for a stage's successor.
///
/// A stage calls it from its own `compute()` and emits the planner it returns, so a constructor
/// failure fails that compute. Constructors only store their input: substantial planning belongs
/// in the successor's `compute()`.
pub type Next<Input> = Arc<dyn Fn(Input) -> VortexResult<Box<dyn Planner>> + Send + Sync>;

/// Builds a `Next` from a closure that returns a concrete planner.
pub fn next_fn<Input, F, P>(f: F) -> Next<Input>
where
    F: Fn(Input) -> VortexResult<P> + Send + Sync + 'static,
    P: Planner + 'static,
{
    Arc::new(move |input: Input| f(input).map(|planner| Box::new(planner) as Box<dyn Planner>))
}

#[cfg(test)]
mod tests {
    use vortex_error::vortex_bail;
    use vortex_io::request::IoConsumer;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoResult;

    use super::*;
    use crate::planning::planner::PlannerOutput;
    use crate::planning::planner::State;

    struct Finished;

    impl IoConsumer for Finished {
        fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
    }

    impl Planner for Finished {
        fn state(&self) -> State {
            State::Done
        }

        fn compute(&mut self) -> VortexResult<PlannerOutput> {
            Ok(PlannerOutput::Done)
        }
    }

    #[test]
    fn next_fn_boxes_the_planner() -> VortexResult<()> {
        let next: Next<u32> = next_fn(|_input: u32| Ok(Finished));
        assert_eq!(next(7)?.state(), State::Done);
        Ok(())
    }

    #[test]
    fn next_fn_reports_constructor_errors() {
        let next: Next<u32> = next_fn(|input: u32| -> VortexResult<Finished> {
            vortex_bail!("constructor rejected {input}")
        });
        let err = next(7).err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("constructor rejected 7")),
            "{err:?}"
        );
    }
}
