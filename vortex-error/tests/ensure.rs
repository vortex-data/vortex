// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks error macro behavior from a caller outside the crate.

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vortex_error::VortexError;
    use vortex_error::VortexErrorKind;
    use vortex_error::VortexResult;

    /// Asserts the error's kind and the message shown after its kind prefix.
    fn assert_error(error: &VortexError, kind: VortexErrorKind, message: &str) {
        assert_eq!(error.kind(), kind, "{error}");
        let shown = error.to_string();
        let body = shown
            .split_once(": ")
            .map_or(shown.as_str(), |(_, body)| body);
        assert!(
            body == message || body.starts_with(&format!("{message}\nBacktrace:")),
            "{shown:?}"
        );
    }

    fn check_default(left: &Cell<u32>, right: &Cell<u32>) -> VortexResult<()> {
        vortex_error::vortex_ensure_eq!(
            left.replace(left.get() + 1),
            right.replace(right.get() + 1),
        );
        Ok(())
    }

    #[test]
    fn default_success_evaluates_each_operand_once() -> VortexResult<()> {
        let left = Cell::new(1);
        let right = Cell::new(1);

        check_default(&left, &right)?;

        assert_eq!(left.get(), 2);
        assert_eq!(right.get(), 2);
        Ok(())
    }

    #[test]
    fn default_failure_reports_the_values_compared() {
        let left = Cell::new(1);
        let right = Cell::new(2);

        let error = check_default(&left, &right).unwrap_err();

        assert_eq!(left.get(), 2);
        assert_eq!(right.get(), 3);
        assert_error(
            &error,
            VortexErrorKind::AssertionFailed,
            "`left.replace(left.get() + 1) == right.replace(right.get() + 1)`\n  left: 1\n right: 2",
        );
    }

    #[test]
    fn custom_message_is_lazy_and_operands_are_borrowed() -> VortexResult<()> {
        let left = String::from("left");
        let right = String::from("right");
        let message_calls = Cell::new(0);
        let context = "field";
        let check = |right: &String| -> VortexResult<()> {
            vortex_error::vortex_ensure_eq!(
                &left,
                right,
                InvalidArgument: "{context} mismatch {}",
                message_calls.replace(message_calls.get() + 1),
            );
            Ok(())
        };

        check(&left)?;
        assert_eq!(message_calls.get(), 0);

        let error = check(&right).unwrap_err();

        assert_eq!(message_calls.get(), 1);
        assert_error(
            &error,
            VortexErrorKind::InvalidArgument,
            "field mismatch 0\n  left: left\n right: right",
        );
        assert_eq!(left, "left");
        assert_eq!(right, "right");
        Ok(())
    }

    #[test]
    fn explicit_variant_keeps_context_and_values() {
        let check = || -> VortexResult<()> {
            vortex_error::vortex_ensure_eq!(1, 2, InvalidArgument: "expected {} fields", 2,);
            Ok(())
        };

        let error = check().unwrap_err();

        assert_error(
            &error,
            VortexErrorKind::InvalidArgument,
            "expected 2 fields\n  left: 1\n right: 2",
        );
    }

    #[test]
    fn kind_without_message_reports_the_values_compared() {
        let check = || -> VortexResult<()> {
            vortex_error::vortex_ensure_eq!(1 + 1, 3, OutOfBounds);
            Ok(())
        };

        let error = check().unwrap_err();

        assert_error(
            &error,
            VortexErrorKind::OutOfBounds,
            "`1 + 1 == 3`\n  left: 2\n right: 3",
        );
    }

    #[test]
    fn qualified_ensure_does_not_require_a_macro_import() {
        let check = || -> VortexResult<()> {
            vortex_error::vortex_ensure!(false);
            Ok(())
        };

        let error = check().unwrap_err();

        assert_error(&error, VortexErrorKind::AssertionFailed, "false");
    }
}
