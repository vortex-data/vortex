// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dispatch macro used by the optimizer and executor to emit trace events.
//!
//! See [`test_harness::trace`][crate::test_harness::trace] for the harness that consumes the
//! events recorded here.

/// Dispatch a trace event from inside the optimizer or executor.
///
/// This macro is the only call-site interface for the trace harness. It is designed so that
/// release builds (and CodSpeed benchmark builds) compile every invocation away to nothing,
/// while debug-test builds with an active recorder route the call into
/// [`test_harness::trace`][crate::test_harness::trace].
///
/// # Forms
///
/// ```ignore
/// // 1. Event dispatch — the common form. Calls `test_harness::trace::<event>(...)` only when
/// //    a recorder is installed on this thread. Otherwise compiles to nothing.
/// crate::trace_op!(record_execute_until_start::<M>(&array));
///
/// // 2. `use(...)` — keep otherwise-unused bindings alive in non-test builds so the
/// //    surrounding code still type-checks when the trace call is compiled out.
/// crate::trace_op!(use(plugin_idx, slot_idx));
///
/// // 3. `value(<active_expr>, <inactive_expr>)` — pick between two expressions depending on
/// //    whether a recorder is active. Useful when the active branch needs to clone state
/// //    just for the trace.
/// let snapshot = crate::trace_op!(value(Some(parent.clone()), None));
///
/// // 4. `scope(<phase>, || <body>)` — execute `body` while a static phase label is pushed
/// //    onto the thread-local stack consulted by `parent_kernel`/`reduce_parent` recorders.
/// //    Equivalent to running `body` directly when tracing is disabled.
/// crate::trace_op!(scope("stack_execute_parent", || run_parent_kernel(...)))
/// ```
///
/// # Gating
///
/// Each invocation expands to a `cfg`-gated branch:
///
/// - When `test` OR `_test-harness` is on, AND `codspeed` is off, the body is compiled in and
///   guarded by a runtime check for an active recorder.
/// - In all other configurations, the body is replaced with `()`. Captured names are bound
///   through the `use(...)` form so the compiler does not warn about unused variables.
///
/// The macro is named `trace_op` because it records *operations* (optimizer rewrites, parent
/// kernels, execution steps, builder activity) rather than array values. The events it emits
/// describe what work the optimizer/executor did, not the contents of any array.
macro_rules! trace_op {
    (@when_enabled { $($enabled:tt)* } else { $($disabled:tt)* }) => {{
        #[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
        {
            $($enabled)*
        }

        #[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
        {
            $($disabled)*
        }
    }};

    (@if_active { $($enabled:tt)* } else { $($disabled:tt)* }) => {
        $crate::trace_op!(@when_enabled {
            if $crate::test_harness::trace::is_active() {
                $($enabled)*
            } else {
                $($disabled)*
            }
        } else {
            $($disabled)*
        })
    };

    (use($($value:expr),* $(,)?)) => {
        $crate::trace_op!(@when_enabled {} else {
            let _ = ($(&$value),*);
        })
    };

    (value($enabled:expr, $disabled:expr)) => {
        $crate::trace_op!(@if_active { $enabled } else { $disabled })
    };

    (scope($phase:expr, || $body:expr)) => {
        $crate::trace_op!(@when_enabled {
            $crate::test_harness::trace::with_execute_parent_phase_if_active($phase, || $body)
        } else {{
            let _ = $phase;
            $body
        }})
    };

    ($($event:tt)*) => {
        $crate::trace_op!(@if_active {
            $crate::test_harness::trace::$($event)*
        } else {})
    };
}

pub(crate) use trace_op;
