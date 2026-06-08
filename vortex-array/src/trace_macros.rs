// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dispatch macro used by the optimizer and executor to emit trace events.
//!
//! See [`test_harness::trace`][crate::test_harness::trace] for the harness that consumes the
//! events recorded here.

#[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
use crate::test_harness::trace::with_execute_parent_phase_if_active;

/// Dispatch a trace event from inside the optimizer or executor.
///
/// This macro is the only call-site interface for the trace harness. It is designed so that
/// release builds (and CodSpeed benchmark builds) compile every invocation away to nothing,
/// while debug-test builds with an active recorder route the call into
/// [`test_harness::trace`][crate::test_harness::trace].
///
/// # Gating
///
/// Each invocation expands to a `cfg`-gated branch:
///
/// - When `test` OR `_test-harness` is on, AND `codspeed` is off, the body is compiled in and
///   guarded by a runtime check for an active recorder.
/// - In all other configurations, the body is replaced with `()`.
///
/// The macro is named `trace_op` because it records *operations* (optimizer rewrites, parent
/// kernels, execution steps, builder activity) rather than array values. The events it emits
/// describe what work the optimizer/executor did, not the contents of any array.
macro_rules! trace_op {
    ($event:ident $(::<$($generic:ty),*>)?($($arg:expr),* $(,)?)) => {{
        #[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
        {
            if $crate::test_harness::trace::is_active() {
                $crate::test_harness::trace::$event $(::<$($generic),*>)?($($arg),*);
            }
        }
    }};
}

#[inline]
pub(crate) fn with_execute_parent_trace_phase<R>(phase: &'static str, f: impl FnOnce() -> R) -> R {
    #[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
    {
        with_execute_parent_phase_if_active(phase, f)
    }

    #[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
    {
        let _ = phase;
        f()
    }
}

pub(crate) use trace_op;
