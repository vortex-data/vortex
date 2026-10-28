// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This seems to be a bug in the lint - https://github.com/rust-lang/rust-clippy/issues/11024
#![expect(clippy::tests_outside_test_module)]

use std::error::Error;
use std::io;
use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::vortex_err;

#[test]
fn test_message_is_displayed_as_written() {
    assert!(
        vortex_err!("index {} past {}", 5, 3)
            .to_string()
            .starts_with("index 5 past 3")
    );
}

#[test]
fn test_with_context_preserves_source() {
    let err = VortexError::from(io::Error::other("disk on fire")).with_context("while reading");

    assert!(
        err.source()
            .and_then(|source| source.downcast_ref::<io::Error>())
            .is_some(),
        "context must not hide the underlying error"
    );

    let display = err.to_string();
    assert!(
        display.starts_with("while reading:\n  disk on fire"),
        "{display}"
    );
}

#[test]
fn test_nested_context() {
    let err = vortex_err!("bad footer")
        .with_context("reading layout")
        .with_context("opening file");

    let display = err.to_string();
    assert!(
        display.starts_with("opening file:\n  reading layout:\n  bad footer"),
        "{display}"
    );
}

#[test]
fn test_external_preserves_concrete_source() {
    let err = vortex_err!(External: io::Error::from(io::ErrorKind::NotFound));

    assert_eq!(
        err.source()
            .and_then(|source| source.downcast_ref::<io::Error>())
            .map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );
}

#[test]
fn test_clone_and_shared_round_trip() {
    let err = vortex_err!("device disappeared");
    let clone = err.clone();

    assert_eq!(clone.to_string(), err.to_string());

    // An `Arc` is unwrapped when uniquely held and cloned otherwise; both keep the message.
    let shared = Arc::new(err);
    let borrowed = VortexError::from(&shared);
    let owned = VortexError::from(shared);

    assert_eq!(borrowed.to_string(), owned.to_string());
}

#[test]
fn test_context_on_a_clone_leaves_the_other_clone_alone() {
    let err = VortexError::from(io::Error::other("disk on fire"));
    let clone = err.clone();

    // The payload is shared, so this context cannot be written into it in place.
    let with_context = clone.with_context("while reading");

    let original = err.to_string();
    assert!(
        original.starts_with("disk on fire") && !original.contains("while reading"),
        "{original}"
    );
    let display = with_context.to_string();
    assert!(
        display.starts_with("while reading:\n  disk on fire"),
        "{display}"
    );
    assert!(
        with_context
            .source()
            .and_then(|source| source.downcast_ref::<io::Error>())
            .is_some(),
        "context on a shared error must still expose the underlying error"
    );
}

#[test]
fn test_arrow_error_is_kept_as_the_source() {
    let err = VortexError::from(arrow_schema::ArrowError::ComputeError("x".into()));
    assert!(
        err.source()
            .and_then(|source| source.downcast_ref::<arrow_schema::ArrowError>())
            .is_some(),
        "the Arrow error is kept as the source"
    );
}
