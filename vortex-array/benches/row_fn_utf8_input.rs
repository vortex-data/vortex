// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures reduce values below 26 before narrowing"
)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::Utf8Column;
use vortex_array::scalar_fn::unstable::row::execute_rows;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

const ROWS: usize = 1 << 14;

/// Short values fit inside a string view. Long values reference a data buffer.
const VALUE_LENS: &[usize] = &[8, 64];

/// The input layouts [`Utf8Column`] decodes differently.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Offsets into one byte buffer.
    VarBin,

    /// Canonical string views.
    VarBinView,
}

const LAYOUTS: &[Layout] = &[Layout::VarBin, Layout::VarBinView];

#[derive(Clone, Copy)]
struct Case {
    len: usize,
    layout: Layout,
}

impl Display for Case {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Preserve the report names from the former const-length and layout parameters.
        let layout = match self.layout {
            Layout::VarBin => "VarBin",
            Layout::VarBinView => "VarBinView",
        };

        write!(f, "{}, {layout}", self.len)
    }
}

fn cases() -> Vec<Case> {
    VALUE_LENS
        .iter()
        .flat_map(|&len| LAYOUTS.iter().map(move |&layout| Case { len, layout }))
        .filter(|case| {
            // VarBin varies on unchanged code, except for short values on the NEON leg.
            env!("VORTEX_BENCH_VARIANT") == "local"
                || matches!(case.layout, Layout::VarBinView)
                || (env!("VORTEX_BENCH_VARIANT") == "neon" && case.len == 8)
        })
        .collect()
}

#[derive(Clone)]
struct StartsWith;

impl RowFn for StartsWith {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_utf8_input.starts_with");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(Utf8Column,), bool>(|(value,)| value.starts_with("abc"))
    }
}

/// Builds `ROWS` ASCII values of `len` bytes, a third of which start with the matched prefix.
fn make_input(layout: Layout, len: usize) -> ArrayRef {
    let values = (0..ROWS).map(|row| {
        let prefix = if row % 3 == 0 { "abc" } else { "xyz" };
        let suffix = (prefix.len()..len).map(|index| char::from(b'a' + ((row + index) % 26) as u8));
        prefix.chars().chain(suffix).collect::<String>()
    });

    match layout {
        Layout::VarBin => {
            VarBinArray::from_iter_nonnull(values, DType::Utf8(Nullability::NonNullable))
                .into_array()
        }
        Layout::VarBinView => VarBinViewArray::from_iter_str(values).into_array(),
    }
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = cases())]
fn starts_with(bencher: Bencher, case: Case) {
    let args = VecExecutionArgs::new(vec![make_input(case.layout, case.len)], ROWS);

    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&args, SESSION.create_execution_ctx()))
        .bench_refs(|(args, ctx)| {
            execute_rows(&StartsWith, &EmptyOptions, *args, ctx)
                .vortex_expect("row execution should succeed in benchmark")
        });
}
