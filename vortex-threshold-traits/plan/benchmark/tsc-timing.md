# TSC Timing for Low-Overhead Measurement

## Problem

`std::time::Instant::now()` has ~20-50ns overhead per call. For fast functions (nanosecond-scale), this timer overhead can dominate or distort measurements.

## Solution: CPU Timestamp Counters

Modern CPUs provide low-overhead timestamp counters:

| Platform | Instruction | Overhead | Notes |
|----------|-------------|----------|-------|
| x86_64 | `RDTSC` | ~1-2ns | Read Time-Stamp Counter |
| aarch64 | `CNTVCT_EL0` | ~1-2ns | Counter-timer Virtual Count |
| Other | `Instant::now()` | ~20-50ns | Fallback |

This is what Divan uses internally for its timing.

## Implementation Sketch

```rust
mod timestamp {
    use std::time::Instant;

    /// Reads the CPU timestamp counter.
    #[inline]
    pub fn now() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // RDTSC: Read Time-Stamp Counter
            // SAFETY: RDTSC is always available on x86_64
            unsafe { core::arch::x86_64::_rdtsc() }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // CNTVCT_EL0: Counter-timer Virtual Count register
            let cnt: u64;
            // SAFETY: Reading CNTVCT_EL0 is always safe on AArch64
            unsafe {
                core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt, options(nostack, nomem));
            }
            cnt
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use Instant
            static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
            let start = START.get_or_init(Instant::now);
            start.elapsed().as_nanos() as u64
        }
    }

    /// Calibrates TSC frequency (ns per tick).
    /// Called once, cached via OnceLock.
    pub fn calibrate() -> f64 {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            use std::sync::OnceLock;
            static NS_PER_TICK: OnceLock<f64> = OnceLock::new();

            *NS_PER_TICK.get_or_init(|| {
                const CALIBRATION_MS: u64 = 10;
                let duration = std::time::Duration::from_millis(CALIBRATION_MS);

                let start_tsc = now();
                let start_instant = Instant::now();

                while start_instant.elapsed() < duration {
                    core::hint::spin_loop();
                }

                let end_tsc = now();
                let elapsed_ns = start_instant.elapsed().as_nanos() as f64;
                let elapsed_ticks = (end_tsc - start_tsc) as f64;

                elapsed_ns / elapsed_ticks
            })
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { 1.0 } // Fallback already returns nanoseconds
    }

    /// Converts tick delta to nanoseconds.
    #[inline]
    pub fn ticks_to_ns(ticks: u64) -> f64 {
        ticks as f64 * calibrate()
    }
}
```

## Usage in Measurer

Replace `Instant::now()` with `timestamp::now()` in the hot path:

```rust
fn collect_samples<I, O, S, R>(...) -> Vec<f64> {
    // ... setup ...

    // Hot path: use TSC for timing
    let batch_start = timestamp::now();
    for input in &inputs {
        black_box(routine(black_box(input)));
    }
    let batch_end = timestamp::now();

    let per_iter_ns = timestamp::ticks_to_ns(batch_end - batch_start)
                      / iters_per_batch as f64;
    samples.push(per_iter_ns);

    // ... continue ...
}
```

## Calibration Considerations

1. **One-time cost**: Calibration takes ~10ms, done once per process via `OnceLock`
2. **Invariant TSC**: Modern CPUs have invariant TSC (constant frequency regardless of CPU state)
3. **Cross-core consistency**: On most modern CPUs, TSC is synchronized across cores
4. **Virtualization**: May be less accurate in VMs, but still better than Instant

## When This Matters

| Function time | Instant overhead | TSC overhead | Error reduction |
|---------------|------------------|--------------|-----------------|
| 1µs | 2-5% | 0.1-0.2% | ~20x |
| 100ns | 20-50% | 1-2% | ~20x |
| 10ns | 200-500% | 10-20% | ~20x |

For functions taking >1µs, the improvement is marginal. For nanosecond-scale operations (like bit manipulation, simple arithmetic), TSC makes the difference between usable and unusable measurements.

## Alternative: Use Divan Directly

If we don't want to maintain TSC code, alternatives:

1. **Divan as harness**: Use `#[divan::bench]` for developer workflow (but can't get programmatic results)
2. **Criterion with custom measurement**: Criterion allows plugging in custom `Measurement` trait implementations
3. **Accept higher overhead**: For threshold detection, we mostly care about relative performance, so consistent overhead may be acceptable

## Recommendation

For the threshold system specifically:

1. **Short term**: Keep `Instant::now()` - it's simple and the batching already amortizes most overhead
2. **Medium term**: Add TSC support when measuring very fast operations becomes a priority
3. **Long term**: Consider contributing a programmatic API to Divan upstream

The current `Measurer` is already well-designed with proper batching. TSC is an optimization, not a correctness issue.

## References

- [Divan timing source](https://github.com/nvzqz/divan/blob/main/src/time.rs)
- [Intel RDTSC documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=rdtsc)
- [ARM CNTVCT_EL0](https://developer.arm.com/documentation/ddi0595/2021-06/AArch64-Registers/CNTVCT-EL0--Counter-timer-Virtual-Count-register)
- [Rust core::arch intrinsics](https://doc.rust-lang.org/core/arch/index.html)
