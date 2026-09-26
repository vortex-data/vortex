// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This seems to be a bug in the lint - https://github.com/rust-lang/rust-clippy/issues/11024
#![expect(clippy::tests_outside_test_module)]

use std::error::Error;
use std::io;
use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::VortexErrorKind;
use vortex_error::vortex_err;

#[test]
fn test_other_err_is_other() {
    assert_eq!(vortex_err!(Other: "boom").kind(), VortexErrorKind::Other);
}

#[test]
fn test_tagged_err_carries_kind() {
    // Every kind takes the same `Kind: "format", args..` shape.
    let cases: [(VortexError, VortexErrorKind); 8] = [
        (
            vortex_err!(InvalidArgument: "bad {}", 1),
            VortexErrorKind::InvalidArgument,
        ),
        (
            vortex_err!(OutOfBounds: "index {} past {}", 5, 3),
            VortexErrorKind::OutOfBounds,
        ),
        (
            vortex_err!(MismatchedTypes: "want i32, got {}", "u8"),
            VortexErrorKind::MismatchedTypes,
        ),
        (
            vortex_err!(NotImplemented: "take on {}", "Sparse"),
            VortexErrorKind::NotImplemented,
        ),
        (
            vortex_err!(NotFound: "no field {}", "a"),
            VortexErrorKind::NotFound,
        ),
        (
            vortex_err!(Overflow: "{} exceeds u32", 1u64 << 40),
            VortexErrorKind::Overflow,
        ),
        (vortex_err!(Serde: "corrupt footer"), VortexErrorKind::Serde),
        (
            vortex_err!(AssertionFailed: "invariant broken"),
            VortexErrorKind::AssertionFailed,
        ),
    ];
    for (err, kind) in cases {
        assert_eq!(err.kind(), kind, "{err}");
    }
}

#[test]
fn test_kind_prefixes_the_display() {
    assert!(
        vortex_err!(NotFound: "no field a")
            .to_string()
            .starts_with("Not found error: no field a")
    );
    assert!(
        vortex_err!(Overflow: "too big")
            .to_string()
            .starts_with("Overflow error: too big")
    );
}

#[test]
fn test_int_conversion_is_overflow() {
    // `?` on a fallible integer cast must land in Overflow, not Other.
    let err = VortexError::from(u8::try_from(300u32).unwrap_err());
    assert_eq!(err.kind(), VortexErrorKind::Overflow);
}

#[test]
fn test_with_context_preserves_kind_and_source() {
    let err = VortexError::from(io::Error::other("disk on fire")).with_context("while reading");

    assert_eq!(err.kind(), VortexErrorKind::Io);
    assert!(
        err.source()
            .and_then(|source| source.downcast_ref::<io::Error>())
            .is_some(),
        "context must not hide the underlying error"
    );

    let display = err.to_string();
    assert!(
        display.starts_with("IO error: while reading:\n  disk on fire"),
        "{display}"
    );
}

#[test]
fn test_nested_context() {
    let err = vortex_err!(Serde: "bad footer")
        .with_context("reading layout")
        .with_context("opening file");

    assert_eq!(err.kind(), VortexErrorKind::Serde);
    let display = err.to_string();
    assert!(
        display.starts_with("Serde error: opening file:\n  reading layout:\n  bad footer"),
        "{display}"
    );
}

#[test]
fn test_external_preserves_concrete_source() {
    let err = vortex_err!(External: io::Error::from(io::ErrorKind::NotFound));

    assert_eq!(err.kind(), VortexErrorKind::Other);
    assert_eq!(
        err.source()
            .and_then(|source| source.downcast_ref::<io::Error>())
            .map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );
}

#[test]
fn test_clone_and_shared_round_trip() {
    let err = vortex_err!(Io: "device disappeared");
    let clone = err.clone();

    assert_eq!(clone.kind(), err.kind());
    assert_eq!(clone.to_string(), err.to_string());

    // An `Arc` is unwrapped when uniquely held and cloned otherwise; both keep the classification.
    let shared = Arc::new(err);
    let borrowed = VortexError::from(&shared);
    let owned = VortexError::from(shared);

    assert_eq!(borrowed.kind(), VortexErrorKind::Io);
    assert_eq!(owned.kind(), VortexErrorKind::Io);
    assert_eq!(borrowed.to_string(), owned.to_string());
}

#[test]
fn test_arrow_error_is_classified_by_variant() {
    use arrow_schema::ArrowError;

    let cases = [
        (
            ArrowError::NotYetImplemented("x".into()),
            VortexErrorKind::NotImplemented,
        ),
        (
            ArrowError::ArithmeticOverflow("x".into()),
            VortexErrorKind::Overflow,
        ),
        (
            ArrowError::SchemaError("x".into()),
            VortexErrorKind::MismatchedTypes,
        ),
        (
            ArrowError::IoError("x".into(), io::Error::other("x")),
            VortexErrorKind::Io,
        ),
        (ArrowError::IpcError("x".into()), VortexErrorKind::Serde),
        (
            ArrowError::InvalidArgumentError("x".into()),
            VortexErrorKind::InvalidArgument,
        ),
        (ArrowError::ComputeError("x".into()), VortexErrorKind::Other),
    ];
    for (arrow, kind) in cases {
        let err = VortexError::from(arrow);
        assert_eq!(err.kind(), kind, "{err}");
        assert!(
            err.source().is_some(),
            "the Arrow error is kept as the source"
        );
    }
}

#[test]
fn test_jiff_error_is_classified() {
    let no_such_date = jiff::civil::Date::new(2025, 2, 29).unwrap_err();
    assert_eq!(
        VortexError::from(no_such_date).kind(),
        VortexErrorKind::InvalidArgument
    );
}
