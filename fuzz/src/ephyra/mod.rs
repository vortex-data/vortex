// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzz targets as ephyra properties, run by `tests/*_ephyra.rs`.
//!
//! Each target has a plan, a plain-data description of a fuzz case that ephyra can shrink and
//! mutate, which materializes into the input of the matching libFuzzer target, and a `check` that
//! runs that target's checks on it: [`array_ops`] and [`file_io`]. The drivers here add what
//! libFuzzer gives the `cargo fuzz` targets:
//!
//! - [`run_corpus`] is a coverage-guided search over a corpus directory of JSON plans. It
//!   replays the corpus first to establish coverage, then alternates fresh plans with mutants of
//!   plans that reached new code, and writes every plan that reaches new code back to the corpus.
//!   Coverage needs the `ephyra-coverage` feature and a `-C instrument-coverage` build; without
//!   them it is a plain random search that adds nothing to the corpus.
//! - [`check_plan`] saves the smallest failing plan for each failure signature to
//!   `EPHYRA_CRASH_DIR`, so a failure reproduces from its file whatever seed or corpus found it.
//! - [`run_plans`] replays the plans named by `EPHYRA_PLAN`.
//! - [`run_random`] is a plain random search, the fixed-seed check that pull requests run.

pub mod array_ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod file_io;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::fs;
use std::ops::ControlFlow;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::Once;
use std::time::Duration;
use std::time::Instant;

use ephyra::BoxGen;
use ephyra::Config;
use ephyra::Gen;
use ephyra::Run;
use ephyra::Source;
use ephyra::TestResult;
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use vortex_array::dtype::DType;
use vortex_utils::aliases::hash_set::HashSet;

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

/// Keep the smallest failing plan per signature in `EPHYRA_CRASH_DIR`, if it is set.
///
/// Shrinking tests ever smaller candidates, so the file ends with the shrunk plan.
fn save_crash<T: Serialize>(plan: &T, signature: &str) {
    let first = SEEN_SIGNATURES.lock().insert(signature.to_string());
    let Some(dir) = std::env::var_os("EPHYRA_CRASH_DIR").map(PathBuf::from) else {
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

/// Save a corpus entry named by its content, so the same plan is never stored twice.
fn save_entry<T: Serialize>(dir: &Path, plan: &T) -> bool {
    let Ok(json) = serde_json::to_vec(plan) else {
        return false;
    };
    let path = dir.join(format!("{:016x}.json", fnv1a(&json)));
    if path.exists() {
        return false;
    }
    write_atomic(&path, &json)
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

/// Yields the corpus, in order, before handing over to the plan type's own generator.
///
/// Shrinking and mutation always use the plan generator, so shrunk and mutated plans stay in
/// its support.
struct CorpusGen<T> {
    corpus: Mutex<VecDeque<T>>,
    plans: BoxGen<T>,
}

impl<T: Send + Sync> Gen for CorpusGen<T> {
    type Value = T;

    fn generate(&self, g: &mut Source) -> T {
        let next = self.corpus.lock().pop_front();
        next.unwrap_or_else(|| self.plans.generate(g))
    }

    fn shrink(&self, value: &T, visitor: &mut dyn FnMut(T) -> ControlFlow<()>) -> ControlFlow<()> {
        self.plans.shrink(value, visitor)
    }

    fn mutate(&self, value: &T, g: &mut Source) -> T {
        self.plans.mutate(value, g)
    }
}

/// Coverage-guided search over the corpus in `EPHYRA_CORPUS_DIR`; see the module documentation.
///
/// The run's budget, seed and failure count come from the usual `EPHYRA_*` variables, and
/// `EPHYRA_COVERAGE_FILTER` picks the crate whose coverage counts as new.
pub fn run_corpus<T, P>(name: &str, property: P)
where
    T: ephyra::Arbitrary + Clone + Debug + Serialize + DeserializeOwned + Send + Sync,
    P: FnMut(T) -> TestResult,
{
    let Some(corpus_dir) = std::env::var_os("EPHYRA_CORPUS_DIR").map(PathBuf::from) else {
        eprintln!("ephyra: set EPHYRA_CORPUS_DIR to run the corpus search");
        return;
    };
    if let Err(e) = fs::create_dir_all(&corpus_dir) {
        eprintln!("ephyra: cannot create {}: {e}", corpus_dir.display());
        return;
    }
    let corpus = load_plans::<T>(&corpus_dir);
    let replay = corpus.len();
    eprintln!("ephyra: replaying {replay} corpus entries");

    let config = Config::default().name(name).no_regressions();
    let generator = CorpusGen {
        corpus: Mutex::new(corpus.into()),
        plans: ephyra::any::<T>(),
    };
    let mut run = Run::new(&config, generator, property);
    let mut drawn = 0usize;
    let mut mutate = false;
    let mut progress = Progress::new(name);
    while run.wants_more() {
        progress.maybe_print(replay);
        let parent = if drawn >= replay {
            mutate = !mutate;
            if mutate { run.pool_pick() } else { None }
        } else {
            None
        };
        let input = match &parent {
            Some(parent) => run.mutate(parent),
            None => {
                drawn += 1;
                run.draw()
            }
        };
        let Some(input) = input else {
            continue;
        };
        let tested = run.test(&input);
        progress.record(&tested, parent.is_some());
        if tested.passed() && tested.novelty() > 0 {
            let saved = save_entry(&corpus_dir, &input.value);
            progress.novel(tested.novelty(), saved, drawn <= replay);
            run.keep_always(input.value, f64::from(tested.novelty()));
        }
    }
    progress.print(replay);
    eprintln!("ephyra: added {} corpus entries", progress.added);
    finish(run.finish());
}

/// Print the full report, passing or not, then fail the test if the run failed.
pub fn finish(report: ephyra::Report) {
    eprintln!("{report}");
    report.assert_passed();
}

/// Counts for the progress lines a corpus search prints every 30 seconds.
struct Progress<'a> {
    name: &'a str,
    started: Instant,
    last: Instant,
    tests: u64,
    discarded: u64,
    failed: u64,
    mutants: u64,
    novel_inputs: u64,
    novelty: u64,
    added: usize,
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
            mutants: 0,
            novel_inputs: 0,
            novelty: 0,
            added: 0,
        }
    }

    fn record(&mut self, tested: &ephyra::Tested, mutant: bool) {
        self.tests += 1;
        self.mutants += u64::from(mutant);
        self.discarded += u64::from(tested.discarded());
        self.failed += u64::from(tested.failed());
    }

    /// An input reached new code: announce the first hundred new corpus entries, then every
    /// hundredth, so a fresh corpus does not flood the log.
    fn novel(&mut self, novelty: u32, saved: bool, replaying: bool) {
        self.novel_inputs += 1;
        self.novelty += u64::from(novelty);
        if saved && !replaying {
            self.added += 1;
            if self.added <= 100 || self.added.is_multiple_of(100) {
                eprintln!(
                    "ephyra: new coverage (+{novelty} regions) at test {}; corpus entry {} added",
                    self.tests, self.added
                );
            }
        }
    }

    fn maybe_print(&mut self, replay: usize) {
        if self.last.elapsed() >= Self::EVERY {
            self.print(replay);
        }
    }

    fn print(&mut self, replay: usize) {
        self.last = Instant::now();
        let secs = self.started.elapsed().as_secs_f64();
        let rate = if secs > 0.0 {
            self.tests as f64 / secs
        } else {
            0.0
        };
        let seen = SEEN_SIGNATURES.lock().len();
        eprintln!(
            "ephyra: [{}] {:.0}s: {} tests ({rate:.0}/s, {} mutants, {} discarded), \
             {} failing inputs over {seen} signature(s), {} inputs with new coverage \
             (+{} regions), corpus {replay} replayed + {} added",
            self.name,
            secs,
            self.tests,
            self.mutants,
            self.discarded,
            self.failed,
            self.novel_inputs,
            self.novelty,
            self.added,
        );
    }
}

/// Check every plan in the files or directories that `EPHYRA_PLAN` names, separated by `:`.
pub fn run_plans<T, F>(mut check: F)
where
    T: DeserializeOwned + Debug,
    F: FnMut(T) -> TestResult,
{
    let Some(paths) = std::env::var_os("EPHYRA_PLAN") else {
        eprintln!("ephyra: set EPHYRA_PLAN to a plan file or directory to replay it");
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

/// A random search of `tests` plans, printing the full report; `EPHYRA_*` variables override it.
pub fn run_random<T, P>(name: &str, tests: u64, property: P)
where
    T: ephyra::Arbitrary + Clone + Debug,
    P: FnMut(T) -> TestResult,
{
    let config = Config::default().name(name).tests(tests);
    finish(ephyra::check(&config, ephyra::any::<T>(), property));
}
