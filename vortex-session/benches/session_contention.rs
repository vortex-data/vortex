// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Isolates snapshot reads from session-handle refcount traffic.
//! Arguments: `MODEL READERS ITERATIONS [--pin]`. Models: `get`, `read`, `private`,
//! `clone-read`, `read-with-cloner`, and `create`. Linux pinning requires `taskset`.
//! No arguments or `--test` perform a short smoke run.

use std::any::Any;
use std::env;
use std::fs;
use std::hint::black_box;
use std::io;
use std::process::Command;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_error::VortexExpect;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

#[derive(Clone, Debug, Default)]
struct Value(u64);

impl SessionVar for Value {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn session() -> VortexSession {
    VortexSession::empty().with_some(Value(42))
}

fn pin(cpu: usize) -> io::Result<()> {
    let stat = fs::read_to_string("/proc/thread-self/stat")?;
    let tid = stat
        .split_whitespace()
        .next()
        .ok_or_else(|| io::Error::other("missing thread ID"))?;
    let output = Command::new("taskset")
        .args(["-pc", &cpu.to_string(), tid])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(())
}

fn read_loop<const DEFAULT: bool>(session: &VortexSession, iterations: u64) {
    for _ in 0..iterations {
        let value = if DEFAULT {
            session.get::<Value>()
        } else {
            session
                .get_opt::<Value>()
                .vortex_expect("value is registered")
        };
        black_box(value.0);
    }
}

fn work(model: &str, shared: &VortexSession, iterations: u64) {
    match model {
        "get" => read_loop::<true>(shared, iterations),
        "read" | "private" | "read-with-cloner" => read_loop::<false>(shared, iterations),
        "clone-read" => {
            for _ in 0..iterations {
                let local = black_box(shared.clone());
                read_loop::<false>(&local, 1);
            }
        }
        "create" => {
            for _ in 0..iterations {
                black_box(session());
            }
        }
        _ => unreachable!("model validated before starting workers"),
    }
}

fn run(model: &str, readers: usize, iterations: u64, pinned: bool) -> io::Result<()> {
    let shared = session();
    let cloner = model == "read-with-cloner";
    let ready = Arc::new(Barrier::new(readers + 1 + usize::from(cloner)));
    let start = Arc::new(Barrier::new(readers + 1 + usize::from(cloner)));
    let done = Arc::new(Barrier::new(readers + 1));
    let stop = AtomicBool::new(false);
    let (elapsed, clones) = thread::scope(|scope| -> io::Result<_> {
        let mut workers = Vec::with_capacity(readers);
        for cpu in 0..readers {
            let shared = if model == "private" {
                session()
            } else {
                shared.clone()
            };
            let (ready, start, done) = (&ready, &start, &done);
            workers.push(scope.spawn(move || -> io::Result<()> {
                let affinity = if pinned { pin(cpu) } else { Ok(()) };
                work(model, &shared, 16);
                ready.wait();
                start.wait();
                if affinity.is_ok() {
                    work(model, &shared, iterations);
                }
                done.wait();
                affinity
            }));
        }
        let background = cloner.then(|| {
            let shared = &shared;
            let (ready, start, stop) = (&ready, &start, &stop);
            scope.spawn(move || -> io::Result<u64> {
                let affinity = if pinned { pin(readers) } else { Ok(()) };
                ready.wait();
                start.wait();
                let mut clones = 0;
                if affinity.is_ok() {
                    while !stop.load(Ordering::Relaxed) {
                        black_box(shared.clone());
                        clones += 1;
                    }
                }
                affinity.map(|()| clones)
            })
        });
        ready.wait();
        let begin = Instant::now();
        start.wait();
        done.wait();
        let elapsed = begin.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        for worker in workers {
            worker
                .join()
                .map_err(|_| io::Error::other("reader panicked"))??;
        }
        let clones = background
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| io::Error::other("cloner panicked"))?
            })
            .transpose()?
            .unwrap_or(0);
        Ok((elapsed, clones))
    })?;
    let operations = readers as f64 * iterations as f64;
    println!(
        "{{\"model\":\"{model}\",\"readers\":{readers},\"iterations_per_reader\":{iterations},\"pinned\":{pinned},\"elapsed_s\":{elapsed},\"cloner_ops\":{clones},\"ns_per_op\":{},\"million_ops_s\":{}}}",
        elapsed * 1e9 / operations,
        operations / elapsed / 1e6,
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let models = [
        "get",
        "read",
        "private",
        "clone-read",
        "read-with-cloner",
        "create",
    ];
    if args.iter().any(|arg| arg == "--list" || arg == "--help") {
        println!("MODEL READERS ITERATIONS [--pin]\n{}", models.join("\n"));
        return Ok(());
    }
    if args.iter().all(|arg| arg == "--bench") || args.iter().any(|arg| arg == "--test") {
        for model in models {
            run(model, 1, 1000, false)?;
        }
        return Ok(());
    }
    let pinned = args.iter().any(|arg| arg == "--pin");
    let args = args
        .iter()
        .filter(|arg| !matches!(arg.as_str(), "--bench" | "--pin"))
        .collect::<Vec<_>>();
    if args.len() != 3 || !models.contains(&args[0].as_str()) {
        return Err(io::Error::other(
            "expected MODEL READERS ITERATIONS [--pin]",
        ));
    }
    let readers = args[1].parse::<usize>().map_err(io::Error::other)?;
    let iterations = args[2].parse::<u64>().map_err(io::Error::other)?;
    if readers == 0 || iterations == 0 {
        return Err(io::Error::other("readers and iterations must be nonzero"));
    }
    run(args[0], readers, iterations, pinned)
}
