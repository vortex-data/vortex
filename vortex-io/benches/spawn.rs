// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::thread;
use std::time::Duration;

use divan::Bencher;
use futures::future::join_all;

#[divan::bench(args = [1, 10, 100], threads = false, sample_count = 1, sample_size = 1)]
fn tokio_spawn(b: Bencher, work_ms: u64) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let handle = rt.handle().clone();
    b.bench_local(|| {
        // Spawn 1000 tasks that all do 1MS of CPU-blocking work.
        let join_handles: Vec<_> = (0..100)
            .map(|_| handle.spawn(async move { thread::sleep(Duration::from_millis(work_ms)) }))
            .collect();

        rt.block_on(join_all(join_handles));
    });
}

#[divan::bench(args = [1, 10, 100], threads = false, sample_count = 1, sample_size = 1)]
fn tokio_spawn_blocking(b: Bencher, work_ms: u64) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let handle = rt.handle().clone();
    b.bench_local(|| {
        let work_ms = work_ms;
        // Spawn 1000 tasks that all do 1MS of CPU-blocking work.
        let join_handles: Vec<_> = (0..100)
            .map(|_| handle.spawn_blocking(move || thread::sleep(Duration::from_millis(work_ms))))
            .collect();

        rt.block_on(join_all(join_handles));
    });
}

#[divan::bench(args = [1, 10, 100], threads = false, sample_count = 1, sample_size = 1)]
fn tokio_unblock(b: Bencher, work_ms: u64) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let handle = rt.handle().clone();
    b.bench_local(|| {
        // Spawn 1000 tasks that all do 1MS of CPU-blocking work.
        let work_ms = work_ms;
        let join_handles: Vec<_> = (0..100)
            .map(|_| {
                handle.spawn(blocking::unblock(move || {
                    thread::sleep(Duration::from_millis(work_ms.clone()))
                }))
            })
            .collect();

        rt.block_on(join_all(join_handles));
    });
}

#[divan::bench(args = [1, 10, 100], threads = false, sample_size = 1, sample_count = 1)]
fn smol_unblock(b: Bencher, work_ms: u64) {
    let exec = smol::Executor::new();

    b.bench_local(|| {
        // Spawn 1000 tasks that all do 1MS of CPU-blocking work.
        let work_ms = work_ms;
        let mut join_handles = Vec::with_capacity(100);
        exec.spawn_many(
            (0..1000).map(|_| {
                blocking::unblock(move || thread::sleep(Duration::from_millis(work_ms.clone())))
            }),
            &mut join_handles,
        );
        // run the executor to unblock shit
        // We need to do this all on the current thread, so this is bad
        smol::block_on(exec.run(async move {
            for handle in join_handles {
                handle.await;
            }
        }));
    });
}

fn main() {
    divan::main();
}
