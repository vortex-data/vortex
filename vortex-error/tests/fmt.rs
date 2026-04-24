// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This seems to be a bug in the lint - https://github.com/rust-lang/rust-clippy/issues/11024
#![expect(clippy::tests_outside_test_module)]

use serial_test::serial;
use vortex_error::VortexError;
use vortex_error::vortex_err;

#[test]
#[serial]
#[inline(never)]
fn test_basic_display() {
    temp_env::with_var("RUST_BACKTRACE", Some("1"), || {
        let err = vortex_err!("this is bad");
        let display = err.to_string();

        assert!(
            display.contains("Other error: this is bad"),
            "should contain error message"
        );
        assert!(
            display.contains("Backtrace:"),
            "should contain backtrace header"
        );

        #[cfg(not(windows))] // for some reason, the windows backtrace is different
        assert!(
            display.contains("test_basic_display"),
            "backtrace should include test function"
        );
    });
}

#[test]
#[serial]
fn test_from_arrow_with_backtrace() {
    temp_env::with_var("RUST_BACKTRACE", Some("1"), || {
        let arrow_error = arrow_schema::ArrowError::NotYetImplemented(
            "This feature isn't implemented yet".to_string(),
        );

        let vx_error = VortexError::from(arrow_error);
        let display = vx_error.to_string();

        assert!(
            display
                .contains("Arrow error: Not yet implemented: This feature isn't implemented yet"),
            "should contain arrow error message"
        );
        assert!(
            display.contains("Backtrace:"),
            "should contain backtrace header"
        );
    });
}

#[test]
#[serial]
fn test_from_arrow_no_backtrace() {
    // Detect a nextest run, because `Backtrace::capture` caches whether backtraces are enabled
    // and `cargo test` runs tests in the same process, while nextest uses separate processes.
    if std::env::var("NEXTEST_RUN_ID").is_ok() {
        temp_env::with_var("RUST_BACKTRACE", Some("0"), || {
            let arrow_error = arrow_schema::ArrowError::NotYetImplemented(
                "This feature isn't implemented yet".to_string(),
            );

            let vx_error = VortexError::from(arrow_error);
            let display = vx_error.to_string();

            assert!(
                display.contains(
                    "Arrow error: Not yet implemented: This feature isn't implemented yet"
                ),
                "should contain arrow error message"
            );
            assert!(
                !display.contains("Backtrace:"),
                "should not contain backtrace when RUST_BACKTRACE=0"
            );
        });
    }
}
