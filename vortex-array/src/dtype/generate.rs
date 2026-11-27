// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hegel generators for dtype property tests.

use std::sync::Arc;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnionVariants;

const PTYPES: [PType; 11] = [
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::F16,
    PType::F32,
    PType::F64,
];

/// Draws any [`PType`].
pub(crate) fn draw_ptype(tc: &TestCase) -> PType {
    PTYPES[tc.draw(gs::integers::<usize>().max_value(PTYPES.len() - 1))]
}

/// Draws a [`DType`] with at most `depth` levels of nested containers.
///
/// Structs and unions have one to three children. Map keys are non-nullable primitives.
/// Extension dtypes are not drawn, because they need a registered extension.
pub(crate) fn draw_dtype(tc: &TestCase, depth: u8) -> DType {
    let kinds = if depth == 0 { 7 } else { 12 };
    let nullability = Nullability::from(tc.draw(gs::booleans()));
    let child = || draw_dtype(tc, depth - 1);
    let width = || tc.draw(gs::integers::<usize>().min_value(1).max_value(3));

    match tc.draw(gs::integers::<u8>().max_value(kinds - 1)) {
        0 => DType::Null,
        1 => DType::Bool(nullability),
        2 => DType::Primitive(draw_ptype(tc), nullability),
        3 => {
            let precision = tc.draw(gs::integers::<u8>().min_value(1).max_value(76));
            let scale = tc.draw(gs::integers::<i8>().max_value(precision.cast_signed()));
            DType::Decimal(DecimalDType::new(precision, scale), nullability)
        }
        4 => DType::Utf8(nullability),
        5 => DType::Binary(nullability),
        6 => DType::Variant(nullability),
        7 => DType::List(Arc::new(child()), nullability),
        8 => DType::FixedSizeList(
            Arc::new(child()),
            tc.draw(gs::integers::<u32>()),
            nullability,
        ),
        9 => DType::struct_(
            (0..width()).map(|i| (format!("f{i}"), child())),
            nullability,
        ),
        10 => DType::map(
            DType::Primitive(draw_ptype(tc), Nullability::NonNullable),
            child(),
            tc.draw(gs::booleans()),
            nullability,
        )
        .vortex_expect("map keys are non-nullable"),
        _ => {
            let n = width();
            DType::Union(
                UnionVariants::new(
                    (0..n).map(|i| format!("v{i}")).collect(),
                    (0..n).map(|_| child()).collect(),
                )
                .vortex_expect("union variants are distinct"),
                nullability,
            )
        }
    }
}
