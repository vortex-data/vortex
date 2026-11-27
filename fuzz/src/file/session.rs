// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::LazyLock;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex::VortexSessionDefault;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex_array::ArrayEq;
use vortex_array::ArrayRef;
use vortex_array::EqMode;
use vortex_array::ExecutionConfig;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::OpenOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::RUNTIME;
use crate::SESSION;
use crate::enable_fuzz_editions;
use crate::file::FuzzFileAction;
use crate::file::has_duplicate_field_names;
use crate::file::has_nullable_struct;
use crate::file::report_on_panic;

const MASK_BYTES: usize = 40;
const VARIANTS: usize = MASK_BYTES * 8; // fits 320 rules
const MAX_ITERATIONS: usize = 1 << 16;

static REDUCED_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    enable_fuzz_editions(&session);
    session.register(ExecutionConfig {
        max_iterations: MAX_ITERATIONS,
    });
    session
});

struct Kernels {
    names: Vec<String>,
    order: Vec<usize>,
}

// This is quite ugly, but kernels don't have a stable id
static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    let mut visited = Vec::new();
    let mut visit = |kernel: &dyn Debug| {
        let name = format!("{kernel:?}");
        let short = name
            .rsplit_once(": ")
            .map_or(name.as_str(), |(_, kernel)| kernel);
        visited.push(short.trim_end_matches(" }").to_string());
        true
    };
    SESSION.kernels().retain(&mut visit);
    SESSION.aggregate_fns().retain_kernels(&mut visit);
    let mut names = visited.clone();
    names.sort_unstable();
    names.dedup();
    let order = visited
        .iter()
        .map(|name| {
            names
                .binary_search(name)
                .ok()
                .vortex_expect("name is listed")
        })
        .collect();
    Kernels { names, order }
});

thread_local! {
    static CURRENT: Cell<Option<Disabled>> = const { Cell::new(None) };
}

#[derive(Debug)]
pub struct FuzzFileSession {
    pub action: FuzzFileAction,
    pub variants: [Disabled; VARIANTS],
}

impl<'a> Arbitrary<'a> for FuzzFileSession {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let action = u.arbitrary()?;
        let mut variants = [Disabled::only(0); VARIANTS];
        for (index, variant) in variants.iter_mut().enumerate() {
            *variant = if u.is_empty() {
                Disabled::only(index)
            } else {
                u.arbitrary()?
            };
        }
        Ok(Self { action, variants })
    }
}

#[derive(Clone, Copy)]
pub struct Disabled {
    kernels: [u8; MASK_BYTES],
    expression_optimizer: bool,
}

impl<'a> Arbitrary<'a> for Disabled {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            kernels: u.arbitrary()?,
            expression_optimizer: u.arbitrary()?,
        })
    }
}

impl Debug for Disabled {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let kernels = KERNELS
            .names
            .iter()
            .enumerate()
            .filter(|(index, _)| self.drops_index(*index))
            .map(|(_, name)| name)
            .collect::<Vec<_>>();
        f.debug_struct("Disabled")
            .field("expression_optimizer", &self.expression_optimizer)
            .field("kernels", &kernels)
            .finish()
    }
}

impl Disabled {
    fn drops_index(&self, index: usize) -> bool {
        let bit = index % (MASK_BYTES * 8);
        self.kernels[bit / 8] >> (bit % 8) & 1 != 0
    }

    fn only(index: usize) -> Self {
        let mut kernels = [0; MASK_BYTES];
        kernels[index / 8] = 1 << (index % 8);
        Self {
            kernels,
            expression_optimizer: false,
        }
    }

    fn install(&self, session: &VortexSession) {
        let mut order = KERNELS.order.iter();
        let mut keeps = || {
            let index = order.next().vortex_expect("kernel registry changed");
            !self.drops_index(*index)
        };
        session.register(SESSION.kernels().retain(|_| keeps()));
        session.register(SESSION.aggregate_fns().retain_kernels(|_| keeps()));
    }
}

pub fn run_file_session(fuzz: FuzzFileSession) -> VortexResult<bool> {
    let FuzzFileSession { action, variants } = fuzz;
    if has_nullable_struct(action.array.dtype()) || has_duplicate_field_names(action.array.dtype())
    {
        return Ok(false);
    }

    let mut file = Vec::new();
    action
        .write_settings
        .write_options()
        .blocking(&*RUNTIME)
        .write(&mut file, action.array.to_array_iterator())?;
    let file = ByteBuffer::from(file);
    let expected = read(&SESSION, file.clone(), &action, true)?;

    report_on_panic(|| {
        if let Some(disabled) = CURRENT.get() {
            eprintln!("{disabled:#?}");
        }
    });

    for disabled in variants {
        CURRENT.set(Some(disabled));
        disabled.install(&REDUCED_SESSION);
        let actual = read(
            &REDUCED_SESSION,
            file.clone(),
            &action,
            !disabled.expression_optimizer,
        )?;
        compare(&expected, &actual)?;
    }
    CURRENT.set(None);
    Ok(true)
}

fn read(
    session: &VortexSession,
    file: ByteBuffer,
    action: &FuzzFileAction,
    optimize: bool,
) -> VortexResult<Vec<ArrayRef>> {
    let file = session.open_options().open_buffer(file)?;
    let bind = |expr: &Expression| -> VortexResult<_> {
        let bound = expr.bind(file.dtype())?;
        if optimize {
            bound.optimize_recursive()
        } else {
            Ok(bound)
        }
    };
    let projection = bind(&action.projection_expr.clone().unwrap_or_else(root))?;
    let filter = action.filter_expr.as_ref().map(bind).transpose()?;
    let mut ctx = session.create_execution_ctx();
    file.scan()?
        .with_projection(projection)
        .with_some_filter(filter)
        .into_array_iter(&*RUNTIME)?
        .map(|batch| -> VortexResult<ArrayRef> {
            let batch = batch?;
            let canonical = batch
                .clone()
                .execute::<RecursiveCanonical>(&mut ctx)
                .map_err(|error| error.with_context(batch.display_tree().to_string()))?;
            Ok(canonical.0.into_array())
        })
        .collect()
}

fn compare(expected: &[ArrayRef], actual: &[ArrayRef]) -> VortexResult<()> {
    if expected.len() != actual.len() {
        vortex_bail!(
            "full session read {} chunks, reduced session read {}",
            expected.len(),
            actual.len()
        );
    }
    let mut ctx = SESSION.create_execution_ctx();
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if !actual.array_eq(expected, EqMode::Value)
            && !all_non_distinct(actual, expected, &mut ctx)?
        {
            vortex_bail!(
                "chunk {index} differs\nfull session\n{}\nreduced session\n{}",
                expected.display_values(&ctx),
                actual.display_values(&ctx)
            );
        }
    }
    Ok(())
}
