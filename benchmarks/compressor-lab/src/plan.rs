// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expanding a spec into a deterministic plan of tasks, and reading and writing plans.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;

use crate::identity::CodeIdentity;
use crate::key::TaskKey;
use crate::key::stable_digest;
use crate::source::ChunkInput;
use crate::source::ResolvedSource;
use crate::source::SkippedColumn;
use crate::source::resolve_source;
use crate::spec::DTypeClass;
use crate::spec::ParamValue;
use crate::spec::PlanSpec;
use crate::spec::SchemeSelection;
use crate::spec::SearchParams;
use crate::spec::SyntheticPType;

/// The version of the plan file format.
pub const PLAN_FORMAT_VERSION: u32 = 1;

/// The plan manifest file name.
pub const PLAN_FILE: &str = "plan.json";

/// The task list file name, one JSON task per line.
pub const TASKS_FILE: &str = "tasks.jsonl";

/// The compressor preset the baseline tasks record.
const BASELINE_PRESET: &str = "btrblocks-default";

/// The kinds of task a plan lists up front. Later stages expand from these tasks' outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Read a row range and store it as a canonical chunk.
    Chunk,
    /// Search encoding trees for one chunk and store every candidate encoding.
    Search,
    /// Compress one chunk with today's compressor and store its encoding.
    Baseline,
}

impl TaskKind {
    /// The lowercase name used in files and on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::Search => "search",
            Self::Baseline => "baseline",
        }
    }

    /// The wave this kind runs in. Every task in a wave can run in parallel.
    pub fn wave(self) -> u32 {
        match self {
            Self::Chunk => 1,
            Self::Search | Self::Baseline => 2,
        }
    }
}

/// One unit of work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// The task's identity.
    pub key: TaskKey,
    /// What the task does.
    pub kind: TaskKind,
    /// The wave the task runs in.
    pub wave: u32,
    /// Tasks whose outputs this task reads.
    pub deps: Vec<TaskKey>,
    /// What the task reads and how.
    pub input: TaskInput,
}

/// A task's input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskInput {
    /// Where a chunk's rows come from.
    Chunk(ChunkInput),
    /// A search over one chunk.
    Search {
        /// The chunk task.
        chunk: TaskKey,
        /// The chunk's dtype class.
        dtype: DTypeClass,
        /// The source name, for provenance.
        source: String,
        /// The search parameters.
        search: SearchParams,
        /// The digest of the plan's scheme list.
        schemes_digest: String,
    },
    /// Today's compressor on one chunk.
    Baseline {
        /// The chunk task.
        chunk: TaskKey,
        /// The chunk's dtype class.
        dtype: DTypeClass,
        /// The source name, for provenance.
        source: String,
        /// The compressor preset.
        preset: String,
    },
}

/// How a stage's tasks are created.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Expansion {
    /// Listed in the plan.
    Static,
    /// Expanded, deterministically, from the facts written by an earlier stage.
    After {
        /// The stage whose outputs this stage expands from.
        stage: String,
    },
}

/// How a stage's tasks may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    /// Any number of tasks at once, on any machines.
    Full,
    /// A single task.
    Single,
    /// One task at a time per machine, on an isolated machine; machines run independently.
    SequentialPerMachine,
}

/// What a stage writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    /// Deterministic results, upserted by task key. Re-running must reproduce them exactly.
    Fact,
    /// Measurements, appended per (task key, machine, run). Re-running adds samples.
    Observation,
}

/// One stage of the pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    /// The stage name.
    pub name: String,
    /// The wave the stage runs in.
    pub wave: u32,
    /// How the stage's tasks are created.
    pub expansion: Expansion,
    /// How the stage's tasks may run.
    pub parallelism: Parallelism,
    /// What the stage writes.
    pub output: Output,
}

fn stages() -> Vec<Stage> {
    let stage = |name: &str, wave, expansion, parallelism, output| Stage {
        name: name.to_string(),
        wave,
        expansion,
        parallelism,
        output,
    };
    let after = |name: &str| Expansion::After {
        stage: name.to_string(),
    };
    vec![
        stage("chunk", 1, Expansion::Static, Parallelism::Full, Output::Fact),
        stage("search", 2, Expansion::Static, Parallelism::Full, Output::Fact),
        stage("baseline", 2, Expansion::Static, Parallelism::Full, Output::Fact),
        stage("features", 3, after("search"), Parallelism::Full, Output::Fact),
        stage("probes", 3, after("search"), Parallelism::Full, Output::Fact),
        stage("verify", 4, after("search"), Parallelism::Full, Output::Fact),
        stage("timing_plan", 5, after("verify"), Parallelism::Single, Output::Fact),
        stage(
            "time",
            6,
            after("timing_plan"),
            Parallelism::SequentialPerMachine,
            Output::Observation,
        ),
    ]
}

/// The plan manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// The plan file format version.
    pub format_version: u32,
    /// A hash of everything the plan was expanded from.
    pub plan_id: String,
    /// The plan name, from the spec.
    pub name: String,
    /// The code the plan's tasks run with.
    pub identity: CodeIdentity,
    /// The spec, as parsed.
    pub spec: PlanSpec,
    /// The scheme ids the search may use, in registration order.
    pub schemes: Vec<String>,
    /// The digest of `schemes`.
    pub schemes_digest: String,
    /// The sources, resolved and fingerprinted.
    pub sources: Vec<ResolvedSource>,
    /// Columns that matched but were not planned, and why.
    pub skipped: Vec<SkippedColumn>,
    /// Every stage, including the ones expanded later.
    pub stages: Vec<Stage>,
    /// The number of tasks of each kind.
    pub task_counts: BTreeMap<String, u64>,
}

/// A plan and its tasks.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRun {
    /// The manifest.
    pub plan: Plan,
    /// The tasks, sorted by wave, kind and key.
    pub tasks: Vec<Task>,
}

/// The result of writing a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// No plan existed; it was written.
    Created,
    /// The same plan already existed; nothing changed.
    Unchanged,
    /// A different plan existed and was replaced.
    Replaced,
}

/// Selects scheme ids from the registered ones, keeping registration order.
pub fn select_schemes(
    registered: &[String],
    selection: &SchemeSelection,
) -> anyhow::Result<Vec<String>> {
    let known: BTreeSet<&str> = registered.iter().map(String::as_str).collect();
    let named = selection.include.iter().flatten().chain(&selection.exclude);
    for id in named {
        if !known.contains(id.as_str()) {
            bail!(
                "unknown scheme id `{id}`; registered schemes are: {}",
                registered.join(", ")
            );
        }
    }
    Ok(registered
        .iter()
        .filter(|id| {
            selection
                .include
                .as_ref()
                .is_none_or(|include| include.contains(id))
        })
        .filter(|id| !selection.exclude.contains(id))
        .cloned()
        .collect())
}

/// The part of a chunk input that identifies its data. Paths are left out, so moving files
/// keeps keys stable.
#[derive(Serialize)]
#[serde(tag = "source_kind", rename_all = "snake_case")]
enum ChunkIdentity<'a> {
    Parquet {
        source: &'a str,
        file_fingerprint: &'a str,
        column: &'a str,
        arrow_type: &'a str,
        row_start: u64,
        row_end: u64,
    },
    Synthetic {
        source: &'a str,
        generator: &'a str,
        ptype: SyntheticPType,
        params: &'a BTreeMap<String, ParamValue>,
        seed: u64,
        rows: u64,
    },
}

impl<'a> From<&'a ChunkInput> for ChunkIdentity<'a> {
    fn from(input: &'a ChunkInput) -> Self {
        match input {
            ChunkInput::Parquet {
                source,
                file_fingerprint,
                column,
                arrow_type,
                row_start,
                row_end,
                ..
            } => Self::Parquet {
                source,
                file_fingerprint,
                column,
                arrow_type,
                row_start: *row_start,
                row_end: *row_end,
            },
            ChunkInput::Synthetic {
                source,
                generator,
                ptype,
                params,
                seed,
                rows,
                ..
            } => Self::Synthetic {
                source,
                generator,
                ptype: *ptype,
                params,
                seed: *seed,
                rows: *rows,
            },
        }
    }
}

#[derive(Serialize)]
struct ChunkKeyMaterial<'a> {
    identity: &'a CodeIdentity,
    chunk: ChunkIdentity<'a>,
}

#[derive(Serialize)]
struct SearchKeyMaterial<'a> {
    identity: &'a CodeIdentity,
    chunk: &'a TaskKey,
    schemes: &'a [String],
    search: &'a SearchParams,
}

#[derive(Serialize)]
struct BaselineKeyMaterial<'a> {
    identity: &'a CodeIdentity,
    chunk: &'a TaskKey,
    preset: &'a str,
}

#[derive(Serialize)]
struct PlanIdMaterial<'a> {
    format_version: u32,
    identity: &'a CodeIdentity,
    spec: &'a PlanSpec,
    schemes: &'a [String],
    sources: &'a [ResolvedSource],
    skipped: &'a [SkippedColumn],
}

/// Expands a spec into a plan.
///
/// The result depends only on the spec, the files it matches (by content), the identity and the
/// registered schemes, so the same inputs always produce the same plan.
pub fn build_plan(
    spec: &PlanSpec,
    spec_dir: &Path,
    identity: &CodeIdentity,
    registered_schemes: &[String],
) -> anyhow::Result<PlannedRun> {
    let schemes = select_schemes(registered_schemes, &spec.schemes)?;
    let schemes_digest = stable_digest("schemes", &schemes)?;

    let mut sources = Vec::with_capacity(spec.sources.len());
    let mut skipped = Vec::new();
    let mut tasks: BTreeMap<(u32, TaskKind, TaskKey), Task> = BTreeMap::new();

    for source_spec in &spec.sources {
        let resolution = resolve_source(source_spec, spec_dir, spec.chunk_rows, &spec.dtypes)?;
        sources.push(resolution.source);
        skipped.extend(resolution.skipped);

        for chunk in resolution.chunks {
            let chunk_key = TaskKey::derive(
                TaskKind::Chunk.as_str(),
                &ChunkKeyMaterial {
                    identity,
                    chunk: ChunkIdentity::from(&chunk),
                },
            )?;
            let search_key = TaskKey::derive(
                TaskKind::Search.as_str(),
                &SearchKeyMaterial {
                    identity,
                    chunk: &chunk_key,
                    schemes: &schemes,
                    search: &spec.search,
                },
            )?;
            let baseline_key = TaskKey::derive(
                TaskKind::Baseline.as_str(),
                &BaselineKeyMaterial {
                    identity,
                    chunk: &chunk_key,
                    preset: BASELINE_PRESET,
                },
            )?;

            let dtype = chunk.dtype();
            let source = chunk.source().to_string();
            let mut insert = |kind: TaskKind, key: TaskKey, deps: Vec<TaskKey>, input| {
                tasks
                    .entry((kind.wave(), kind, key.clone()))
                    .or_insert(Task {
                        key,
                        kind,
                        wave: kind.wave(),
                        deps,
                        input,
                    });
            };
            insert(
                TaskKind::Search,
                search_key,
                vec![chunk_key.clone()],
                TaskInput::Search {
                    chunk: chunk_key.clone(),
                    dtype,
                    source: source.clone(),
                    search: spec.search.clone(),
                    schemes_digest: schemes_digest.clone(),
                },
            );
            insert(
                TaskKind::Baseline,
                baseline_key,
                vec![chunk_key.clone()],
                TaskInput::Baseline {
                    chunk: chunk_key.clone(),
                    dtype,
                    source,
                    preset: BASELINE_PRESET.to_string(),
                },
            );
            insert(
                TaskKind::Chunk,
                chunk_key,
                Vec::new(),
                TaskInput::Chunk(chunk),
            );
        }
    }

    let tasks: Vec<Task> = tasks.into_values().collect();
    let mut task_counts = BTreeMap::new();
    for task in &tasks {
        *task_counts.entry(task.kind.as_str().to_string()).or_default() += 1;
    }

    let plan_id = stable_digest(
        "plan",
        &PlanIdMaterial {
            format_version: PLAN_FORMAT_VERSION,
            identity,
            spec,
            schemes: &schemes,
            sources: &sources,
            skipped: &skipped,
        },
    )?;

    Ok(PlannedRun {
        plan: Plan {
            format_version: PLAN_FORMAT_VERSION,
            plan_id,
            name: spec.name.clone(),
            identity: identity.clone(),
            spec: spec.clone(),
            schemes,
            schemes_digest,
            sources,
            skipped,
            stages: stages(),
            task_counts,
        },
        tasks,
    })
}

impl PlannedRun {
    /// Writes `plan.json` and `tasks.jsonl` into `dir`.
    ///
    /// Writing the same plan again is a no-op. Writing a different plan over an existing one
    /// fails unless `replace` is set.
    pub fn write(&self, dir: &Path, replace: bool) -> anyhow::Result<WriteOutcome> {
        let plan_bytes = self.plan_bytes()?;
        let task_bytes = self.task_bytes()?;
        let plan_path = dir.join(PLAN_FILE);
        let tasks_path = dir.join(TASKS_FILE);

        let outcome = if plan_path.exists() {
            let existing = fs::read(&plan_path)
                .with_context(|| format!("reading {}", plan_path.display()))?;
            let existing_tasks = fs::read(&tasks_path).unwrap_or_default();
            if existing == plan_bytes && existing_tasks == task_bytes {
                return Ok(WriteOutcome::Unchanged);
            }
            if !replace {
                let existing: Plan = serde_json::from_slice(&existing)
                    .with_context(|| format!("parsing {}", plan_path.display()))?;
                bail!(
                    "{} already holds plan {}, which differs from plan {}; pass --replace to \
                     overwrite it",
                    dir.display(),
                    existing.plan_id,
                    self.plan.plan_id
                );
            }
            WriteOutcome::Replaced
        } else {
            WriteOutcome::Created
        };

        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        write_atomically(&tasks_path, &task_bytes)?;
        write_atomically(&plan_path, &plan_bytes)?;
        Ok(outcome)
    }

    /// Reads a plan written by [`write`](Self::write).
    pub fn read(dir: &Path) -> anyhow::Result<Self> {
        let plan_path = dir.join(PLAN_FILE);
        let plan: Plan = serde_json::from_slice(
            &fs::read(&plan_path).with_context(|| format!("reading {}", plan_path.display()))?,
        )
        .with_context(|| format!("parsing {}", plan_path.display()))?;
        if plan.format_version != PLAN_FORMAT_VERSION {
            bail!(
                "{} has format version {}, but this binary reads version {PLAN_FORMAT_VERSION}",
                plan_path.display(),
                plan.format_version
            );
        }

        let tasks_path = dir.join(TASKS_FILE);
        let text = fs::read_to_string(&tasks_path)
            .with_context(|| format!("reading {}", tasks_path.display()))?;
        let tasks = text
            .lines()
            .enumerate()
            .map(|(line, json)| {
                serde_json::from_str(json)
                    .with_context(|| format!("parsing {} line {}", tasks_path.display(), line + 1))
            })
            .collect::<anyhow::Result<Vec<Task>>>()?;
        Ok(Self { plan, tasks })
    }

    fn plan_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes = serde_json::to_vec_pretty(&self.plan)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn task_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for task in &self.tasks {
            serde_json::to_writer(&mut bytes, task)?;
            bytes.push(b'\n');
        }
        Ok(bytes)
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    let mut file =
        fs::File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing {}", temp.display()))?;
    file.sync_all()?;
    fs::rename(&temp, path).with_context(|| format!("renaming into {}", path.display()))
}

/// The path of a task's done marker in a store.
///
/// A runner writes the marker after the task's outputs are durable, so a task is done exactly
/// when its marker exists.
pub fn ledger_path(store: &Path, key: &TaskKey) -> PathBuf {
    let fan_out = key.as_str().get(..2).unwrap_or("00");
    store
        .join("ledger")
        .join(fan_out)
        .join(format!("{key}.json"))
}
