use std::future::Future;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tokio::runtime::Runtime;

pub fn run<O, R, F>(runtime: &Runtime, iterations: usize, mut routine: R) -> Duration
where
    R: FnMut() -> F,
    F: Future<Output = O>,
{
    run_with_setup(runtime, iterations, || (), |_| routine())
}

pub fn run_with_setup<I, O, S, R, F>(
    runtime: &Runtime,
    iterations: usize,
    mut setup: S,
    mut routine: R,
) -> Duration
where
    S: FnMut() -> I,
    R: FnMut(I) -> F,
    F: Future<Output = O>,
{
    let mut fastest_result = Duration::from_millis(u64::MAX);
    for _ in 0..iterations {
        let state = black_box(setup());
        let elapsed = runtime.block_on(async {
            let start = Instant::now();
            let output = routine(state).await;
            let elapsed = start.elapsed();
            drop(black_box(output));
            elapsed
        });
        fastest_result = fastest_result.min(elapsed);
    }

    fastest_result
}
