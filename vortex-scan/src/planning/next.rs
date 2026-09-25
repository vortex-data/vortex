// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Handoff between stages: pending planners and the `Next` constructor.

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::planning::planner::Planner;

/// Owned, unstarted work that can move to a worker.
///
/// `start()` runs at most once, on the worker that will own the live planner. Substantial
/// planning belongs in the planner's `compute()`, not here.
pub trait PendingPlanner: Send {
    /// Consumes the pending work and constructs the live planner.
    fn start(self: Box<Self>) -> VortexResult<Box<dyn Planner>>;
}

/// A reusable constructor for a stage's successor.
///
/// Calling it packages the input into pending work; it does not run the successor's
/// constructor, so a constructor failure surfaces from `start()`.
pub type Next<Input> = Arc<dyn Fn(Input) -> VortexResult<Box<dyn PendingPlanner>> + Send + Sync>;

struct PendingFn<F>(F);

impl<F> PendingPlanner for PendingFn<F>
where
    F: FnOnce() -> VortexResult<Box<dyn Planner>> + Send + 'static,
{
    fn start(self: Box<Self>) -> VortexResult<Box<dyn Planner>> {
        (self.0)()
    }
}

/// Wraps a `Send + FnOnce` as a pending planner.
pub fn pending<F>(start: F) -> Box<dyn PendingPlanner>
where
    F: FnOnce() -> VortexResult<Box<dyn Planner>> + Send + 'static,
{
    Box::new(PendingFn(start))
}

/// Builds a `Next` from a closure that returns a live planner, deferring the constructor to
/// `start()`.
pub fn next_fn<Input, F, P>(f: F) -> Next<Input>
where
    Input: Send + 'static,
    F: Fn(Input) -> VortexResult<P> + Send + Sync + 'static,
    P: Planner + 'static,
{
    let f = Arc::new(f);
    Arc::new(move |input: Input| {
        let f = Arc::clone(&f);
        Ok(pending(move || {
            f(input).map(|planner| Box::new(planner) as Box<dyn Planner>)
        }))
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    use super::*;
    use vortex_io::request::IoConsumer;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoResult;
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
    fn pending_starts_once() -> VortexResult<()> {
        let starts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&starts);
        let pending = pending(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Finished) as Box<dyn Planner>)
        });
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        let planner = pending.start()?;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(planner.state(), State::Done);
        Ok(())
    }

    #[test]
    fn next_fn_defers_constructor_errors_to_start() -> VortexResult<()> {
        let next: Next<u32> = next_fn(|input: u32| -> VortexResult<Finished> {
            vortex_bail!("constructor rejected {input}")
        });
        let pending = next(7)?;
        let err = pending.start().err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("constructor rejected 7")),
            "{err:?}"
        );
        Ok(())
    }

    struct LocalPlanner(Rc<Cell<bool>>);

    impl IoConsumer for LocalPlanner {
        fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
    }

    impl Planner for LocalPlanner {
        fn state(&self) -> State {
            if self.0.get() {
                State::Done
            } else {
                State::NeedsCompute
            }
        }

        fn compute(&mut self) -> VortexResult<PlannerOutput> {
            self.0.set(true);
            Ok(PlannerOutput::Done)
        }
    }

    #[test]
    fn pending_moves_to_worker_before_constructing_local_state() -> VortexResult<()> {
        let pending = pending(|| Ok(Box::new(LocalPlanner(Rc::new(Cell::new(false))))));
        std::thread::spawn(move || -> VortexResult<()> {
            let mut planner = pending.start()?;
            assert_eq!(planner.state(), State::NeedsCompute);
            assert!(matches!(planner.compute()?, PlannerOutput::Done));
            assert_eq!(planner.state(), State::Done);
            Ok(())
        })
        .join()
        .map_err(|_| vortex_err!("planner worker panicked"))?
    }

}
