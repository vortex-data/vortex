use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use vortex_trace_replay::format::record::TracePayload;
use vortex_trace_replay::{ReplayCursor, TimelinePos, TraceFile, fixture};

#[derive(Parser)]
#[command(name = "vtrace", version, about = "Vortex trace inspector")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the file header, turn count, and timeline summary.
    Inspect { path: PathBuf },

    /// Reconstruct and print the scheduler state at a given turn
    /// (and optional sub-turn event index).
    At {
        path: PathBuf,
        turn: u32,
        #[arg(default_value_t = 0)]
        event_in_turn: u32,
    },

    /// Stream every event as ND-JSON.
    Dump { path: PathBuf },

    /// Print every position whose event matches `--action`.
    Find {
        path: PathBuf,
        #[arg(long)]
        action: String,
    },

    /// Generate a synthetic fixture file (used by tests and the
    /// viewer until the recorder is implemented).
    GenFixture { path: PathBuf },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {}", e);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Cmd::Inspect { path } => cmd_inspect(&path),
        Cmd::At {
            path,
            turn,
            event_in_turn,
        } => cmd_at(&path, turn, event_in_turn),
        Cmd::Dump { path } => cmd_dump(&path),
        Cmd::Find { path, action } => cmd_find(&path, &action),
        Cmd::GenFixture { path } => cmd_gen_fixture(&path),
    }
}

fn cmd_inspect(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = TraceFile::open(path)?;
    let header = file.header();
    let summary = file.timeline_summary();
    println!("file: {}", path.display());
    println!("format_version: {}", header.format_version);
    println!("recorder_version: {}", header.recorder_version);
    println!(
        "task_options: workers={} memory_limit={} max_turns={}",
        header.task_options.worker_count,
        header.task_options.memory_limit_bytes,
        header.task_options.max_turns,
    );
    println!(
        "operators: {}, channels: {}, brokers: {}",
        header.operators.len(),
        header.channels.len(),
        header.brokers.len(),
    );
    println!(
        "timeline: turns={} events={} snapshots={}",
        summary.turns, summary.total_events, summary.total_snapshots
    );
    Ok(())
}

fn cmd_at(
    path: &PathBuf,
    turn: u32,
    event_in_turn: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = TraceFile::open(path)?;
    let mut cursor = ReplayCursor::new(&file);
    cursor.seek(TimelinePos {
        turn,
        event_in_turn,
    })?;
    let json = serde_json::to_string_pretty(cursor.state())?;
    println!("{}", json);
    Ok(())
}

fn cmd_dump(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = TraceFile::open(path)?;
    let mut cursor = ReplayCursor::new(&file);
    cursor.seek(TimelinePos {
        turn: 0,
        event_in_turn: 0,
    })?;
    let mut total = 0u64;
    while let Some(record) = cursor.step_forward()? {
        let json = serde_json::to_string(&record)?;
        println!("{}", json);
        total += 1;
        if total > 100_000 {
            break;
        }
    }
    Ok(())
}

fn cmd_find(path: &PathBuf, action: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = TraceFile::open(path)?;
    let mut cursor = ReplayCursor::new(&file);
    cursor.seek(TimelinePos {
        turn: 0,
        event_in_turn: 0,
    })?;
    while let Some(record) = cursor.step_forward()? {
        if record.payload.variant() == action {
            let pos = cursor.position();
            println!(
                "turn={} event_in_turn={} variant={}",
                pos.turn,
                pos.event_in_turn,
                record.payload.variant(),
            );
            // Print the payload too.
            let payload = serde_json::to_string(&record.payload)?;
            println!("  {}", payload);
            let _ = matches!(record.payload, TracePayload::TurnBegin);
        }
    }
    Ok(())
}

fn cmd_gen_fixture(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    fixture::write_fixture(&mut writer)?;
    println!("wrote synthetic fixture to {}", path.display());
    Ok(())
}
