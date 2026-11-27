// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzz targets as ephyra properties, run by `tests/*_ephyra.rs`.
//!
//! The properties draw their cases from a [`ChoicePlan`] through [`FuzzSource`], the same
//! generators the libFuzzer targets draw from bytes, and each target's `check` runs that target's
//! checks: [`array_ops`] and [`file_io`]. A plan is one number per decision plus the byte strings
//! that whole values such as arrays and scalars decode from, so ephyra shrinks a failure one
//! decision at a time. The drivers:
//!
//! - [`run_random`] is a random search that prints progress every 30 seconds and its full report
//!   at the end. Pull requests run it with a fixed seed, and the scheduled CI search with a time
//!   budget and a fresh seed per worker.
//! - [`check_plan`] saves the smallest failing plan for each failure signature to
//!   `VORTEX_FUZZ_CRASH_DIR`, so a failure reproduces from its file whatever seed found it.
//! - [`run_plans`] replays the plans named by `VORTEX_FUZZ_PLAN`.

pub mod array_ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod file_io;

use std::cell::RefCell;
use std::fmt::Debug;
use std::fs;
use std::ops::RangeInclusive;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::Once;
use std::time::Duration;
use std::time::Instant;

use arbitrary::Unstructured;
use ephyra::Config;
use ephyra::Run;
use ephyra::TestResult;
use ephyra::gens;
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use vortex_array::dtype::DType;
use vortex_utils::aliases::hash_set::HashSet;

use crate::FuzzSource;

/// The decisions of one fuzz case: a number per decision, and the byte strings that whole values
/// decode from. An exhausted plan answers with the smallest choice.
#[derive(Clone, Debug, ephyra::Arbitrary, serde::Serialize, serde::Deserialize)]
pub struct ChoicePlan {
    /// One number per decision; no decision has more than 65,536 outcomes worth telling apart.
    #[ephyra(gen = gens::frequency(vec![
        (3, gens::vec(ephyra::any::<u16>()).len(0..=256).boxed()),
        (1, gens::vec(ephyra::any::<u16>()).len(0..=4096).boxed()),
    ]))]
    pub choices: Vec<u16>,
    /// Mostly short, so that generating and recording a plan stays cheap.
    #[ephyra(gen = gens::vec(gens::frequency(vec![
        (3, gens::vec(gens::int(0u8..=255)).len(0..=64).boxed()),
        (1, gens::vec(gens::int(0u8..=255)).len(0..=4096).boxed()),
    ])).len(0..=6))]
    pub values: Vec<Vec<u8>>,
}

impl ChoicePlan {
    /// A source that reads this plan's choices and values in order.
    pub fn source(&self) -> ChoiceSource<'_> {
        ChoiceSource {
            plan: self,
            choice: 0,
            value: 0,
        }
    }
}

/// Reads a [`ChoicePlan`] as a [`FuzzSource`].
pub struct ChoiceSource<'a> {
    plan: &'a ChoicePlan,
    choice: usize,
    value: usize,
}

impl ChoiceSource<'_> {
    fn next(&mut self) -> u64 {
        let choice = self.plan.choices.get(self.choice).copied().unwrap_or(0);
        self.choice += 1;
        u64::from(choice)
    }
}

impl FuzzSource for ChoiceSource<'_> {
    fn int_in_range(&mut self, range: RangeInclusive<usize>) -> arbitrary::Result<usize> {
        let (start, end) = range.into_inner();
        if start > end {
            return Err(arbitrary::Error::IncorrectFormat);
        }
        let span = u128::from(end as u64 - start as u64) + 1;
        let offset = (u128::from(self.next()) % span) as usize;
        Ok(start + offset)
    }

    fn choose_index(&mut self, len: usize) -> arbitrary::Result<usize> {
        if len == 0 {
            return Err(arbitrary::Error::EmptyChoose);
        }
        self.int_in_range(0..=len - 1)
    }

    fn bool(&mut self) -> arbitrary::Result<bool> {
        Ok(self.next() & 1 == 1)
    }

    fn bytes<R>(
        &mut self,
        decode: impl FnOnce(&mut Unstructured<'_>) -> arbitrary::Result<R>,
    ) -> arbitrary::Result<R> {
        let bytes = self
            .plan
            .values
            .get(self.value)
            .map_or(&[][..], Vec::as_slice);
        self.value += 1;
        decode(&mut Unstructured::new(bytes))
    }
}

/// How one plan fared.
pub enum PlanOutcome {
    /// The checks held.
    Pass,
    /// The plan does not apply; ephyra tries another.
    Discard(&'static str),
    /// A check failed. Failures with the same signature are one bug.
    Fail { signature: String, message: String },
}

/// A saved failure: the plan and the signature it failed with.
#[derive(serde::Serialize, serde::Deserialize)]
struct Crash<T> {
    signature: String,
    plan: T,
}

thread_local! {
    static PANIC_LOCATION: RefCell<Option<String>> = const { RefCell::new(None) };
}

static PANIC_HOOK: Once = Once::new();

/// Failure signatures this process has seen, so each is announced once.
static SEEN_SIGNATURES: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// Remember where the last panic on this thread happened, then hand over to the hook before.
///
/// Installed from inside a property, after ephyra has installed its own hook, so ephyra still
/// records the panic for its report.
fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let location = info.location().map(|l| {
                format!(
                    "{}:{}:{}",
                    l.file().replace('\\', "/"),
                    l.line(),
                    l.column()
                )
            });
            PANIC_LOCATION.with(|slot| *slot.borrow_mut() = location);
            previous(info);
        }));
    });
}

/// Run `check` on `plan` and turn its outcome into ephyra's, saving failing plans.
///
/// A panic is re-raised for ephyra to report, after the plan is saved under the panic location,
/// which is also ephyra's signature for it.
pub fn check_plan<T: Serialize>(plan: &T, check: impl FnOnce() -> PlanOutcome) -> TestResult {
    install_panic_hook();
    match panic::catch_unwind(AssertUnwindSafe(check)) {
        Ok(PlanOutcome::Pass) => TestResult::Pass,
        Ok(PlanOutcome::Discard(reason)) => TestResult::discard(reason),
        Ok(PlanOutcome::Fail { signature, message }) => {
            save_crash(plan, &signature);
            ephyra::signature(signature);
            TestResult::fail(message)
        }
        Err(payload) => {
            let location = PANIC_LOCATION.with(|slot| slot.borrow_mut().take());
            save_crash(plan, location.as_deref().unwrap_or("unknown panic"));
            panic::resume_unwind(payload)
        }
    }
}

/// FNV-1a, a stable hash for file names (std's hasher may change between releases).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Write `bytes` to `path` through a temporary file and a rename, so workers sharing a directory
/// never read a partly written file.
fn write_atomic(path: &Path, bytes: &[u8]) -> bool {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let written = fs::write(&tmp, bytes).and_then(|()| fs::rename(&tmp, path));
    if let Err(e) = written {
        eprintln!("ephyra: cannot write {}: {e}", path.display());
        drop(fs::remove_file(&tmp));
        return false;
    }
    true
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    match serde_json::to_vec(value) {
        Ok(json) => {
            write_atomic(path, &json);
        }
        Err(e) => eprintln!("ephyra: cannot serialize {}: {e}", path.display()),
    }
}

/// Keep the smallest failing plan per signature in `VORTEX_FUZZ_CRASH_DIR`, if it is set.
///
/// Shrinking tests ever smaller candidates, so the file ends with the shrunk plan.
fn save_crash<T: Serialize>(plan: &T, signature: &str) {
    let first = SEEN_SIGNATURES.lock().insert(signature.to_string());
    let Some(dir) = std::env::var_os("VORTEX_FUZZ_CRASH_DIR").map(PathBuf::from) else {
        if first {
            eprintln!("ephyra: NEW FAILURE `{signature}`");
        }
        return;
    };
    if first {
        eprintln!(
            "ephyra: NEW FAILURE `{signature}` (smallest plan kept in {})",
            dir.display()
        );
    }
    let Ok(plan_json) = serde_json::to_vec(plan) else {
        return;
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("ephyra: cannot create {}: {e}", dir.display());
        return;
    }
    let path = dir.join(format!("crash-{:016x}.json", fnv1a(signature.as_bytes())));
    let smaller = match fs::read(&path) {
        Ok(existing) => serde_json::from_slice::<Crash<serde_json::Value>>(&existing)
            .ok()
            .and_then(|crash| serde_json::to_vec(&crash.plan).ok())
            .is_none_or(|existing| plan_json.len() < existing.len()),
        Err(_) => true,
    };
    if smaller {
        let crash = Crash {
            signature: signature.to_string(),
            plan,
        };
        write_json(&path, &crash);
    }
}

/// Every plan in `dir` that parses, in file-name order. Crash files hold their plan in `plan`.
fn load_plans<T: DeserializeOwned>(dir: &Path) -> Vec<T> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.iter().filter_map(|path| load_plan(path)).collect()
}

fn load_plan<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = fs::read(path)
        .map_err(|e| eprintln!("ephyra: cannot read {}: {e}", path.display()))
        .ok()?;
    if let Ok(crash) = serde_json::from_slice::<Crash<T>>(&bytes) {
        return Some(crash.plan);
    }
    serde_json::from_slice::<T>(&bytes)
        .map_err(|e| eprintln!("ephyra: skipping {}: {e}", path.display()))
        .ok()
}

/// Print the full report, passing or not, then fail the test if the run failed.
pub fn finish(report: ephyra::Report) {
    eprintln!("{report}");
    report.assert_passed();
}

/// Counts for the progress lines a search prints every 30 seconds.
struct Progress<'a> {
    name: &'a str,
    started: Instant,
    last: Instant,
    tests: u64,
    discarded: u64,
    failed: u64,
}

impl<'a> Progress<'a> {
    const EVERY: Duration = Duration::from_secs(30);

    fn new(name: &'a str) -> Self {
        let now = Instant::now();
        Self {
            name,
            started: now,
            last: now,
            tests: 0,
            discarded: 0,
            failed: 0,
        }
    }

    fn record(&mut self, tested: &ephyra::Tested) {
        self.tests += 1;
        self.discarded += u64::from(tested.discarded());
        self.failed += u64::from(tested.failed());
        if self.last.elapsed() >= Self::EVERY {
            self.print();
        }
    }

    fn print(&mut self) {
        self.last = Instant::now();
        let secs = self.started.elapsed().as_secs_f64();
        let rate = if secs > 0.0 {
            self.tests as f64 / secs
        } else {
            0.0
        };
        let seen = SEEN_SIGNATURES.lock().len();
        eprintln!(
            "ephyra: [{}] {secs:.0}s: {} tests ({rate:.0}/s, {} discarded), {} failing inputs \
             over {seen} signature(s)",
            self.name, self.tests, self.discarded, self.failed,
        );
    }
}

/// Check every plan in the files or directories that `VORTEX_FUZZ_PLAN` names, separated by `:`.
pub fn run_plans<T, F>(mut check: F)
where
    T: DeserializeOwned + Debug,
    F: FnMut(T) -> TestResult,
{
    let Some(paths) = std::env::var_os("VORTEX_FUZZ_PLAN") else {
        eprintln!("ephyra: set VORTEX_FUZZ_PLAN to a plan file or directory to replay it");
        return;
    };
    let mut failed = Vec::new();
    for path in std::env::split_paths(&paths) {
        let plans = if path.is_dir() {
            load_plans::<T>(&path)
        } else {
            load_plan::<T>(&path).into_iter().collect()
        };
        for plan in plans {
            eprintln!("ephyra: checking {}:\n{plan:?}", path.display());
            if let TestResult::Fail(message) = check(plan) {
                failed.push(format!("{}: {message}", path.display()));
            }
        }
    }
    assert!(failed.is_empty(), "plans failed:\n{}", failed.join("\n"));
}

/// A short name for a dtype's kind, for labels and failure signatures.
pub fn dtype_kind(dtype: &DType) -> String {
    match dtype {
        DType::Null => "null".into(),
        DType::Bool(_) => "bool".into(),
        DType::Primitive(ptype, _) => ptype.to_string(),
        DType::Decimal(..) => "decimal".into(),
        DType::Utf8(_) => "utf8".into(),
        DType::Binary(_) => "binary".into(),
        DType::List(..) => "list".into(),
        DType::FixedSizeList(..) => "fixed_size_list".into(),
        DType::Struct(..) => "struct".into(),
        DType::Map(..) => "map".into(),
        DType::Union(..) => "union".into(),
        DType::Variant(_) => "variant".into(),
        DType::Extension(_) => "extension".into(),
    }
}

/// A random search of `tests` plans; `EPHYRA_*` variables override the count, seed and budget.
pub fn run_random<T, P>(name: &str, tests: u64, property: P)
where
    T: ephyra::Arbitrary + Clone + Debug,
    P: FnMut(T) -> TestResult,
{
    let config = Config::default().name(name).tests(tests);
    let mut run = Run::new(&config, ephyra::any::<T>(), property);
    let mut progress = Progress::new(name);
    if !run.remembered() {
        while run.wants_more() {
            let Some(input) = run.draw() else {
                continue;
            };
            let tested = run.test(&input);
            progress.record(&tested);
        }
    }
    progress.print();
    finish(run.finish());
}
