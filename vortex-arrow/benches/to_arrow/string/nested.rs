// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::ItemsCount;
use itertools::iproduct;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;

use super::STRING_ROWS;
use super::StringEncoding;
use super::StringValidity;
use super::encode_strings;
use super::filtered;
use super::sliced;
use super::structured_strings;
use super::take;
use crate::SESSION;

#[derive(Clone, Copy)]
enum NestedOperator {
    TakeFilter,
    FilterTake,
    SliceFilter,
    FilterSlice,
}

impl Display for NestedOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TakeFilter => "take_filter",
            Self::FilterTake => "filter_take",
            Self::SliceFilter => "slice_filter",
            Self::FilterSlice => "filter_slice",
        })
    }
}

#[derive(Clone, Copy)]
struct NestedStringCase {
    encoding: StringEncoding,
    operators: NestedOperator,
    validity: StringValidity,
}

impl Display for NestedStringCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.encoding, self.operators, self.validity)
    }
}

const ENCODINGS: &[StringEncoding] = &[
    StringEncoding::Fsst,
    StringEncoding::OnPair,
    StringEncoding::Zstd,
];
const OPERATORS: &[NestedOperator] = &[
    NestedOperator::TakeFilter,
    NestedOperator::FilterTake,
    NestedOperator::SliceFilter,
    NestedOperator::FilterSlice,
];
const VALIDITIES: &[StringValidity] = &[StringValidity::NonNullable, StringValidity::Nullable];

fn nested_string_cases() -> Vec<NestedStringCase> {
    iproduct!(
        ENCODINGS.iter().copied(),
        OPERATORS.iter().copied(),
        VALIDITIES.iter().copied()
    )
    .map(|(encoding, operators, validity)| NestedStringCase {
        encoding,
        operators,
        validity,
    })
    .collect()
}

fn apply_nested_operators(array: ArrayRef, operators: NestedOperator) -> ArrayRef {
    match operators {
        NestedOperator::TakeFilter => take(filtered(array)),
        NestedOperator::FilterTake => filtered(take(array)),
        NestedOperator::SliceFilter => sliced(filtered(array)),
        NestedOperator::FilterSlice => filtered(sliced(array)),
    }
}

fn nested_string_array(case: NestedStringCase) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = encode_strings(
        structured_strings(STRING_ROWS, case.validity),
        case.encoding,
        &mut ctx,
    );
    apply_nested_operators(array, case.operators)
}

/// Measures offset array export through two lazy operators.
#[divan::bench(args = nested_string_cases())]
fn nested_string_export(bencher: Bencher, case: NestedStringCase) {
    let array = nested_string_array(case);
    let field = Field::new("value", DataType::Utf8, array.dtype().is_nullable());

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            SESSION
                .arrow()
                .execute_arrow(array, Some(&field), &mut ctx)
                .unwrap()
        });
}
