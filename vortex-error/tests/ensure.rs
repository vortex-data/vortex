// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks error macro behavior from a caller outside the crate.

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vortex_error::VortexError;
    use vortex_error::VortexResult;

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
        let VortexError::AssertionFailed(message, _) = error else {
            panic!("expected AssertionFailed, got {error}");
        };
        assert_eq!(
            message.as_ref(),
            "`left.replace(left.get() + 1) == right.replace(right.get() + 1)`\n  left: 1\n right: 2"
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
                "{context} mismatch {}",
                message_calls.replace(message_calls.get() + 1),
            );
            Ok(())
        };

        check(&left)?;
        assert_eq!(message_calls.get(), 0);

        let error = check(&right).unwrap_err();

        assert_eq!(message_calls.get(), 1);
        let VortexError::Other(message, _) = error else {
            panic!("expected Other, got {error}");
        };
        assert_eq!(
            message.as_ref(),
            "field mismatch 0\n  left: left\n right: right"
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

        let VortexError::InvalidArgument(message, _) = error else {
            panic!("expected InvalidArgument, got {error}");
        };
        assert_eq!(message.as_ref(), "expected 2 fields\n  left: 1\n right: 2");
    }

    #[test]
    fn qualified_ensure_does_not_require_a_macro_import() {
        let check = || -> VortexResult<()> {
            vortex_error::vortex_ensure!(false);
            Ok(())
        };

        let error = check().unwrap_err();

        let VortexError::AssertionFailed(message, _) = error else {
            panic!("expected AssertionFailed, got {error}");
        };
        assert_eq!(message.as_ref(), "false");
    }
}
