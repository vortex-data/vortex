// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures RowFn UTF-8 decoding, including validation and null-view sanitation.

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::Utf8Column;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const SIZES: &[usize] = &[8_192];
const CASES: &[(StringLayout, ValidityPattern)] = &[
    (StringLayout::Inline, ValidityPattern::AllValid),
    (StringLayout::External, ValidityPattern::AllValid),
    (StringLayout::Inline, ValidityPattern::OneNullInEight),
    (StringLayout::External, ValidityPattern::OneNullInEight),
];

#[derive(Clone, Copy, Debug)]
enum StringLayout {
    Inline,
    External,
}

#[derive(Clone, Copy, Debug)]
enum ValidityPattern {
    AllValid,
    OneNullInEight,
}

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = CASES, consts = SIZES)]
fn decode<const ROWS: usize>(
    bencher: Bencher,
    &(layout, validity): &(StringLayout, ValidityPattern),
) {
    let value = match layout {
        StringLayout::Inline => "hello",
        StringLayout::External => "a longer external UTF-8 string λ",
    };
    let array = match validity {
        ValidityPattern::AllValid => {
            VarBinViewArray::from_iter_str(std::iter::repeat_n(value, ROWS))
        }
        ValidityPattern::OneNullInEight => VarBinViewArray::from_iter_nullable_str(
            (0..ROWS).map(|index| (!index.is_multiple_of(8)).then_some(value)),
        ),
    }
    .into_array();

    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            Utf8Column::decode((*array).clone(), ctx)
                .vortex_expect("benchmark strings must decode as UTF-8")
        });
}
