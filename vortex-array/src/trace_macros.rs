// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

macro_rules! trace_array {
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
        $crate::trace_array!(@when_enabled {
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
        $crate::trace_array!(@when_enabled {} else {
            let _ = ($(&$value),*);
        })
    };

    (value($enabled:expr, $disabled:expr)) => {
        $crate::trace_array!(@if_active { $enabled } else { $disabled })
    };

    (scope($phase:expr, || $body:expr)) => {
        $crate::trace_array!(@when_enabled {
            $crate::test_harness::trace::with_execute_parent_phase_if_active($phase, || $body)
        } else {{
            let _ = $phase;
            $body
        }})
    };

    ($($event:tt)*) => {
        $crate::trace_array!(@if_active {
            $crate::test_harness::trace::$($event)*
        } else {})
    };
}

pub(crate) use trace_array;
