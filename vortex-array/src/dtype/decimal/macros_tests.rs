// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::panic::catch_unwind;

use num_traits::AsPrimitive;

use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;

const TYPES: [DecimalType; 6] = [
    DecimalType::I8,
    DecimalType::I16,
    DecimalType::I32,
    DecimalType::I64,
    DecimalType::I128,
    DecimalType::I256,
];

// Only widening pairs implement this trait. Using it in the macro bodies makes instantiating
// even one narrowing combination a compile error, independently of the runtime test inputs.
trait Widen<Out> {
    fn widen(self) -> Out;
}

macro_rules! impl_widen {
    ($out:ty; $( $input:ty ),+) => {
        $(
            impl Widen<$out> for $input {
                fn widen(self) -> $out {
                    self.as_()
                }
            }
        )+
    };
}

impl_widen!(i8; i8);
impl_widen!(i16; i8, i16);
impl_widen!(i32; i8, i16, i32);
impl_widen!(i64; i8, i16, i32, i64);
impl_widen!(i128; i8, i16, i32, i64, i128);
impl_widen!(i256; i8, i16, i32, i64, i128, i256);

#[test]
fn unary_dispatches_only_widening_pairs() {
    for input in TYPES {
        for output in TYPES {
            let result = catch_unwind(|| {
                match_decimal_unary!(input, output, |In, Out| {
                    let value: In = (-42i8).as_();
                    let widened: Out = value.widen();
                    assert_eq!(In::DECIMAL_TYPE, input);
                    assert_eq!(Out::DECIMAL_TYPE, output);
                    widened.to_string()
                })
            });
            if input <= output {
                assert_eq!(result.unwrap(), "-42");
            } else {
                assert!(result.is_err());
            }
        }
    }
}

#[test]
fn binary_dispatches_only_widening_triples() {
    for left in TYPES {
        for right in TYPES {
            for output in TYPES {
                let result = catch_unwind(|| {
                    match_decimal_binary!(left, right, output, |Left, Right, Out| {
                        let lhs: Left = (-42i8).as_();
                        let rhs: Right = 43i8.as_();
                        let lhs: Out = lhs.widen();
                        let rhs: Out = rhs.widen();
                        assert_eq!(Left::DECIMAL_TYPE, left);
                        assert_eq!(Right::DECIMAL_TYPE, right);
                        assert_eq!(Out::DECIMAL_TYPE, output);
                        (lhs + rhs).to_string()
                    })
                });
                if left <= output && right <= output {
                    assert_eq!(result.unwrap(), "1");
                } else {
                    assert!(result.is_err());
                }
            }
        }
    }
}

#[test]
fn dispatch_evaluates_type_expressions_once_in_order() {
    let calls = Cell::new(0);
    let next = |expected| {
        assert_eq!(calls.get(), expected);
        calls.set(expected + 1);
        DecimalType::I8
    };
    match_decimal_unary!(next(0), next(1), |In, Out| {
        assert_eq!(In::DECIMAL_TYPE, Out::DECIMAL_TYPE);
    });
    match_decimal_binary!(next(2), next(3), next(4), |Left, Right, Out| {
        assert_eq!(Left::DECIMAL_TYPE, Out::DECIMAL_TYPE);
        assert_eq!(Right::DECIMAL_TYPE, Out::DECIMAL_TYPE);
    });
    assert_eq!(calls.get(), 5);
}
