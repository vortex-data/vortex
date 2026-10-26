// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public byte-width dispatch, including fallback and control-flow behavior.

use rstest::rstest;
use vortex_buffer::match_each_power_of_two_width;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

fn dispatch_with_error(width: usize, value: VortexResult<usize>) -> VortexResult<usize> {
    let result = match_each_power_of_two_width!(
        width,
        |W| { value? + W },
        other => { vortex_bail!("unsupported record width {other}") },
    );
    Ok(result)
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(4)]
#[case(8)]
#[case(16)]
#[case(32)]
fn specializes_record_width(#[case] width: usize) {
    let records = match_each_power_of_two_width!(
        width,
        |W| { vec![[7u8; W]; 2].into_iter().flatten().collect::<Vec<_>>() },
        _ => { panic!("expected a specialized width") },
    );
    assert_eq!(records, vec![7; width * 2]);
}

#[rstest]
#[case(0)]
#[case(3)]
#[case(12)]
#[case(64)]
fn preserves_unsupported_width(#[case] width: usize) {
    let result = match_each_power_of_two_width!(
        width,
        |W| { Ok(W) },
        other => { Err(other) },
    );
    assert_eq!(result, Err(width));
}

#[rstest]
#[case(2, Err(2))]
#[case(8, Ok(8))]
#[case(64, Ok(64))]
fn restricts_specializations(#[case] width: usize, #[case] expected: Result<usize, usize>) {
    let result = match_each_power_of_two_width!(
        width,
        widths = [8, 64],
        |W| { Ok(W) },
        other => { Err(other) },
    );
    assert_eq!(result, expected);
}

#[test]
fn evaluates_input_and_selected_body_once() {
    let mut input_calls = 0;
    let mut body_calls = 0;
    let value = String::from("selected");
    let result = match_each_power_of_two_width!(
        {
            input_calls += 1;
            4
        },
        |W| {
            body_calls += 1;
            (W, value)
        },
        _ => { panic!("expected a specialized width") },
    );

    assert_eq!((input_calls, body_calls), (1, 1));
    assert_eq!(result, (4, String::from("selected")));
}

#[test]
fn propagates_errors_to_enclosing_function() -> VortexResult<()> {
    assert_eq!(dispatch_with_error(8, Ok(3))?, 11);
    assert!(dispatch_with_error(8, Err(vortex_err!("payload error"))).is_err());
    assert!(dispatch_with_error(3, Ok(3)).is_err());
    Ok(())
}
