// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
macro_rules! trace_array {
    ($($event:tt)*) => {
        if $crate::test_harness::trace::is_active() {
            $crate::test_harness::trace::$($event)*
        }
    };
}

#[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
macro_rules! trace_array {
    ($($event:tt)*) => {{}};
}

#[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        if $crate::test_harness::trace::is_active() {
            $enabled
        } else {
            $disabled
        }
    };
}

#[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        $disabled
    };
}

#[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
macro_rules! trace_array_scope {
    ($phase:expr, || $body:expr) => {
        $crate::test_harness::trace::with_execute_parent_phase_if_active($phase, || $body)
    };
}

#[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
macro_rules! trace_array_scope {
    ($phase:expr, || $body:expr) => {{
        let _ = $phase;
        $body
    }};
}

#[cfg(all(any(test, feature = "_test-harness"), not(codspeed)))]
macro_rules! trace_array_use {
    ($($value:expr),* $(,)?) => {{}};
}

#[cfg(any(not(any(test, feature = "_test-harness")), codspeed))]
macro_rules! trace_array_use {
    ($($value:expr),* $(,)?) => {
        let _ = ($(&$value),*);
    };
}

pub(crate) use trace_array;
pub(crate) use trace_array_scope;
pub(crate) use trace_array_use;
pub(crate) use trace_array_value;
