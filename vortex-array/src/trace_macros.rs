// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array {
    ($($event:tt)*) => {
        $crate::test_harness::trace::if_active(
            || $crate::test_harness::trace::$($event)*,
            || {},
        )
    };
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array {
    ($($event:tt)*) => {{}};
}

#[cfg(any(test, feature = "_test-harness"))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        $crate::test_harness::trace::if_active(|| $enabled, || $disabled)
    };
}

#[cfg(not(any(test, feature = "_test-harness")))]
macro_rules! trace_array_value {
    ($enabled:expr, $disabled:expr) => {
        $disabled
    };
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
pub(crate) use trace_array_use;
pub(crate) use trace_array_value;
