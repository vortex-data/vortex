// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array {
    ($($event:tt)*) => {
        if $crate::test_harness::trace::is_active() {
            $crate::test_harness::trace::$($event)*
        }
    };
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array {
    ($($event:tt)*) => {{}};
}

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        if $crate::test_harness::trace::is_active() {
            $enabled
        } else {
            $disabled
        }
    };
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        $disabled
    };
}

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array_scope {
    ($phase:expr, || $body:expr) => {
        $crate::test_harness::trace::with_execute_parent_phase_if_active($phase, || $body)
    };
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array_scope {
    ($phase:expr, || $body:expr) => {{
        let _ = $phase;
        $body
    }};
}

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array_use {
    ($($value:expr),* $(,)?) => {{}};
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array_use {
    ($($value:expr),* $(,)?) => {
        let _ = ($(&$value),*);
    };
}

pub(crate) use trace_array;
pub(crate) use trace_array_scope;
pub(crate) use trace_array_use;
pub(crate) use trace_array_value;
