// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Training data and evaluation for the learned scheme selector in `vortex_compressor::selector`.
//!
//! [`gather`] compresses each integer chunk with production while recording every site's
//! features, then forces each eligible scheme (and canonical) at each site in turn, with
//! production everywhere else, and measures the whole chunk. Each row is one site and one
//! candidate, so `train_selector.py` sees every loser's cost, which regret weighting needs.
//!
//! It also times production against replaying production's own plan. Replaying skips every
//! estimate, so the gap is the most any selector can save: the "sampling share" ceiling.
//!
//! [`evaluate`] compresses the same chunks with production and with the learned policy, and
//! reports bytes, decode and compression time per chunk.

use std::fmt::Write as _;
use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionPolicy;
use vortex_btrblocks::FollowPlan;
use vortex_btrblocks::SchemeId;
use vortex_btrblocks::SchemeSelection;
use vortex_btrblocks::SiteOutcome;
use vortex_btrblocks::SiteRequest;
use vortex_btrblocks::selector::INT_FEATURE_NAMES;
use vortex_btrblocks::selector::RecordFeatures;
use vortex_btrblocks::selector::TreeModel;

/// One integer chunk to learn from.
pub struct Chunk {
    /// The leave-one-out group, e.g. a dataset or a TPC-H table.
    pub group: String,
    /// The column name.
    pub column: String,
    /// The chunk's index within its column.
    pub index: usize,
    /// The canonical array.
    pub array: ArrayRef,
}

impl Chunk {
    /// A stable id for joining the output tables.
    pub fn id(&self) -> String {
        format!("{}/{}/{}", self.group, self.column, self.index)
    }
}

/// The integer chunks of the given TPC-H tables at a scale factor.
pub fn tpch_chunks(
    tables: &[&str],
    scale_factor: f64,
    chunk_rows: usize,
    max_chunks_per_column: usize,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    for table in tables {
        let batches = crate::tpch::table(table, scale_factor)?;
        let schema = batches[0].schema();
        for (index, field) in schema.fields().iter().enumerate() {
            if !field.data_type().is_integer() {
                continue;
            }
            let columns: Vec<&dyn arrow_array::Array> =
                batches.iter().map(|b| b.column(index).as_ref()).collect();
            let column = arrow_select::concat::concat(&columns)?;
            let n = column.len().div_ceil(chunk_rows).min(max_chunks_per_column);
            for i in 0..n {
                let start = i * chunk_rows;
                let len = chunk_rows.min(column.len() - start);
                let slice = column.slice(start, len);
                chunks.push(Chunk {
                    group: format!("tpch.{table}"),
                    column: format!("{table}.{}", field.name()),
                    index: i,
                    array: crate::load::to_vortex(&slice, ctx)?,
                });
            }
        }
    }
    Ok(chunks)
}

/// Forces one selection at one site and estimates everywhere else, recording what was applied
/// there.
#[derive(Debug)]
struct ForceAt {
    history: Vec<(SchemeId, usize)>,
    selection: SchemeSelection,
    applied: Mutex<Option<Option<SchemeId>>>,
}

impl CompressionPolicy for ForceAt {
    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        if request.ctx.cascade_history() == self.history.as_slice() {
            self.selection.clone()
        } else {
            SchemeSelection::Estimate
        }
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        if outcome.ctx.cascade_history() == self.history.as_slice() {
            *self.applied.lock() = Some(outcome.scheme);
        }
    }
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn time_decode(array: &ArrayRef, reps: u32, ctx: &mut ExecutionCtx) -> anyhow::Result<u128> {
    black_box(array.clone().execute::<Canonical>(ctx)?);
    let mut times = Vec::with_capacity(reps as usize);
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        black_box(array.clone().execute::<Canonical>(ctx)?);
        times.push(start.elapsed().as_nanos());
    }
    Ok(median(times))
}

fn time_compress(
    compressor: &BtrBlocksCompressor,
    array: &ArrayRef,
    reps: u32,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<(ArrayRef, u128)> {
    let out = compressor.compress(array, ctx)?;
    let mut times = Vec::with_capacity(reps as usize);
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        black_box(compressor.compress(array, ctx)?);
        times.push(start.elapsed().as_nanos());
    }
    Ok((out, median(times)))
}

fn production(session: &VortexSession, policy: Option<Arc<dyn CompressionPolicy>>) -> BtrBlocksCompressor {
    let builder = BtrBlocksCompressorBuilder::from_session(session);
    match policy {
        Some(policy) => builder.with_policy(policy),
        None => builder,
    }
    .build()
}

/// How to gather and evaluate.
#[derive(Debug, Clone)]
pub struct Options {
    /// Repetitions for compression timing.
    pub compress_reps: u32,
    /// Repetitions for decode timing.
    pub decode_reps: u32,
    /// Skip forcing candidates at sites deeper than this.
    pub max_depth: usize,
}

/// Writes `chunks.csv`, `sites.csv` and `candidates.csv` into `out`.
pub fn gather(
    session: &VortexSession,
    chunks: &[Chunk],
    options: &Options,
    out: &Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let mut ctx = session.create_execution_ctx();
    let mut chunk_csv = BufWriter::new(File::create(out.join("chunks.csv"))?);
    writeln!(
        chunk_csv,
        "chunk,group,column,rows,canonical_bytes,bytes,decode_ns,compress_ns,replay_ns"
    )?;
    let mut site_csv = BufWriter::new(File::create(out.join("sites.csv"))?);
    writeln!(
        site_csv,
        "chunk,group,column,path,applied,{}",
        INT_FEATURE_NAMES.join(",")
    )?;
    let mut cand_csv = BufWriter::new(File::create(out.join("candidates.csv"))?);
    writeln!(
        cand_csv,
        "chunk,path,scheme,applied,bytes,decode_ns,compress_ns"
    )?;
    let base = production(session, None);

    for (n, chunk) in chunks.iter().enumerate() {
        let id = chunk.id();
        let canonical_bytes = chunk.array.nbytes();
        let (compressed, compress_ns) =
            time_compress(&base, &chunk.array, options.compress_reps, &mut ctx)?;
        let decode_ns = time_decode(&compressed, options.decode_reps, &mut ctx)?;
        let (_, plan) = base.compress_recording_plan(&chunk.array, &mut ctx)?;
        let replay = production(session, Some(Arc::new(FollowPlan::new(plan))));
        let (_, replay_ns) = time_compress(&replay, &chunk.array, options.compress_reps, &mut ctx)?;
        writeln!(
            chunk_csv,
            "{id},{},{},{},{canonical_bytes},{},{decode_ns},{compress_ns},{replay_ns}",
            chunk.group,
            chunk.column,
            chunk.array.len(),
            compressed.nbytes()
        )?;

        let recorder = Arc::new(RecordFeatures::new());
        production(session, Some(Arc::<RecordFeatures>::clone(&recorder) as _))
            .compress(&chunk.array, &mut ctx)?;
        let sites = recorder.take();
        for site in &sites {
            let mut row = format!(
                "{id},{},{},{},{}",
                chunk.group,
                chunk.column,
                site.path,
                site.applied.map_or("canonical".to_string(), |s| s.to_string())
            );
            for f in site.features {
                write!(row, ",{f}")?;
            }
            writeln!(site_csv, "{row}")?;
            if site.history.len() > options.max_depth {
                continue;
            }
            let mut selections: Vec<(String, SchemeSelection)> = site
                .eligible
                .iter()
                .map(|id| (id.to_string(), SchemeSelection::Scheme(*id)))
                .collect();
            selections.push(("canonical".to_string(), SchemeSelection::Canonical));
            for (name, selection) in selections {
                let policy = Arc::new(ForceAt {
                    history: site.history.clone(),
                    selection,
                    applied: Mutex::new(None),
                });
                let forced = production(session, Some(Arc::<ForceAt>::clone(&policy) as _));
                let (out, ns) = time_compress(&forced, &chunk.array, 1, &mut ctx)?;
                let applied = match *policy.applied.lock() {
                    Some(Some(s)) => s.to_string(),
                    Some(None) => "canonical".to_string(),
                    None => "unreached".to_string(),
                };
                let dec = time_decode(&out, options.decode_reps, &mut ctx)?;
                writeln!(
                    cand_csv,
                    "{id},{},{name},{applied},{},{dec},{ns}",
                    site.path,
                    out.nbytes()
                )?;
            }
        }
        eprintln!("[{}/{}] {id}: {} sites", n + 1, chunks.len(), sites.len());
    }
    chunk_csv.flush()?;
    site_csv.flush()?;
    cand_csv.flush()?;
    Ok(())
}

/// Counts what the learned policy decided.
#[derive(Debug, Default)]
struct Decisions {
    picked: usize,
    fallback: usize,
}

/// The learned policy, counting its decisions.
#[derive(Debug)]
struct Counting {
    inner: vortex_btrblocks::selector::LearnedPolicy,
    decisions: Mutex<Decisions>,
}

impl CompressionPolicy for Counting {
    fn select_scheme_with_stats(
        &self,
        request: &SiteRequest<'_>,
        data: &vortex_btrblocks::ArrayAndStats,
        exec_ctx: &mut ExecutionCtx,
    ) -> SchemeSelection {
        let selection = self.inner.select_scheme_with_stats(request, data, exec_ctx);
        let mut d = self.decisions.lock();
        match &selection {
            SchemeSelection::Scheme(_) => d.picked += 1,
            _ => d.fallback += 1,
        }
        selection
    }
}

/// Replays a plan, but reads the site's integer statistics first, as the learned policy does.
#[derive(Debug)]
struct ReplayWithStats(FollowPlan);

impl CompressionPolicy for ReplayWithStats {
    fn select_scheme_with_stats(
        &self,
        request: &SiteRequest<'_>,
        data: &vortex_btrblocks::ArrayAndStats,
        exec_ctx: &mut ExecutionCtx,
    ) -> SchemeSelection {
        if let vortex_array::dtype::DType::Primitive(p, _) = request.array.dtype()
            && p.is_int()
        {
            black_box(data.integer_stats(exec_ctx));
        }
        self.0.select_scheme(request)
    }
}

/// Writes `eval.csv` into `out`: production against the learned policy for each chunk.
pub fn evaluate(
    session: &VortexSession,
    chunks: &[Chunk],
    model: &'static TreeModel,
    min_confidence: f64,
    fallback_k: usize,
    options: &Options,
    out: &Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let mut ctx = session.create_execution_ctx();
    let mut csv = BufWriter::new(File::create(out.join("eval.csv"))?);
    writeln!(
        csv,
        "chunk,group,column,canonical_bytes,base_bytes,base_decode_ns,base_compress_ns,\
         model_bytes,model_decode_ns,model_compress_ns,picked_sites,fallback_sites,\
         replay_ns,replay_stats_ns"
    )?;
    let base = production(session, None);
    let counting = Arc::new(Counting {
        inner: vortex_btrblocks::selector::LearnedPolicy::new(model)
            .with_min_confidence(min_confidence)
            .with_fallback_k(fallback_k),
        decisions: Mutex::new(Decisions::default()),
    });
    let learned = production(session, Some(Arc::<Counting>::clone(&counting) as _));
    for chunk in chunks {
        let (b, b_ns) = time_compress(&base, &chunk.array, options.compress_reps, &mut ctx)?;
        let b_dec = time_decode(&b, options.decode_reps, &mut ctx)?;
        *counting.decisions.lock() = Decisions::default();
        let m = learned.compress(&chunk.array, &mut ctx)?;
        let (picked, fallback) = {
            let d = counting.decisions.lock();
            (d.picked, d.fallback)
        };
        let (_, m_ns) = time_compress(&learned, &chunk.array, options.compress_reps, &mut ctx)?;
        let m_dec = time_decode(&m, options.decode_reps, &mut ctx)?;
        let (_, plan) = base.compress_recording_plan(&chunk.array, &mut ctx)?;
        let replay = production(session, Some(Arc::new(FollowPlan::new(plan.clone()))));
        let (_, r_ns) = time_compress(&replay, &chunk.array, options.compress_reps, &mut ctx)?;
        let replay = production(
            session,
            Some(Arc::new(ReplayWithStats(FollowPlan::new(plan)))),
        );
        let (_, rs_ns) = time_compress(&replay, &chunk.array, options.compress_reps, &mut ctx)?;
        writeln!(
            csv,
            "{},{},{},{},{},{b_dec},{b_ns},{},{m_dec},{m_ns},{picked},{fallback},{r_ns},{rs_ns}",
            chunk.id(),
            chunk.group,
            chunk.column,
            chunk.array.nbytes(),
            b.nbytes(),
            m.nbytes()
        )?;
    }
    csv.flush()?;
    Ok(())
}
