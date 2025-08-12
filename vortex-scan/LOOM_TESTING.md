# Loom Concurrency Testing for vortex-scan

## Overview

The vortex-scan crate includes comprehensive loom tests to verify the correctness of concurrent code. Loom is a concurrency testing tool that exhaustively tests all possible thread interleavings to find race conditions, deadlocks, and other concurrency bugs.

## Running Loom Tests

To run the loom tests, use the following command:

```bash
# Run all loom tests
RUSTFLAGS="--cfg loom" cargo test --release --test loom_concurrency

# Run a specific loom test
RUSTFLAGS="--cfg loom" cargo test --release --test loom_concurrency test_work_queue_atomic_ordering
```

**Important:** Always run loom tests in release mode (`--release`) for better performance. Loom tests can be slow as they exhaustively check all possible interleavings.

## Test Coverage

The loom tests cover the following critical concurrent components:

### 1. Work Queue Atomic Ordering (`test_work_queue_atomic_ordering`)
- **Tests:** The atomic ordering fix in `work_queue.rs:192`
- **Bug Fixed:** Race condition where workers could miss factory updates due to incorrect `Relaxed` ordering
- **Verification:** Ensures `Acquire` ordering properly synchronizes with `Release` ordering

### 2. Work Stealing (`test_work_stealing_no_double_processing`)
- **Tests:** Work stealing pattern doesn't process items twice
- **Verification:** Ensures no duplicate processing when multiple workers steal from the same queue

### 3. Filter RwLock Access (`test_filter_rwlock_concurrent_access`)
- **Tests:** Concurrent access to selectivity histograms in `filter.rs`
- **Verification:** Multiple readers and writers can safely access different RwLocks without deadlock

### 4. Dynamic Version TOCTOU Fix (`test_dynamic_version_toctou_fix`)
- **Tests:** The TOCTOU race condition fix in `tasks.rs:130`
- **Bug Fixed:** Race where version could change between check and use
- **Verification:** Reading version once prevents inconsistent state

### 5. Factory Construction (`test_concurrent_factory_construction`)
- **Tests:** Concurrent factory construction in work queue
- **Verification:** Multiple threads can safely construct factories without races

### 6. Stealer Registration (`test_stealer_registration_race`)
- **Tests:** RwLock pattern for stealer registration
- **Verification:** Safe concurrent registration of work stealers

## Understanding Loom Output

When a loom test fails, it will provide:
1. The specific interleaving that caused the failure
2. A backtrace showing where the issue occurred
3. Details about the type of failure (data race, deadlock, etc.)

Example output:
```
thread 'test_work_queue_atomic_ordering' panicked at:
assertion failed: seen_factories == 3
note: loom found issue after 1234 iterations
```

## Performance Considerations

- Loom tests can take significant time (seconds to minutes per test)
- The number of iterations grows exponentially with thread count and synchronization points
- Use `loom::model()` with small thread counts (2-3) for faster tests
- Consider using `LOOM_MAX_THREADS` environment variable to limit thread count

## Adding New Loom Tests

When adding concurrent code to vortex-scan:

1. Identify critical synchronization points
2. Write a minimal loom test that exercises those points
3. Keep test scope small - test one specific behavior
4. Use descriptive test names that indicate what's being tested

Example template:
```rust
#[test]
fn test_my_concurrent_feature() {
    loom::model(|| {
        // Setup shared state
        let shared = Arc::new(AtomicUsize::new(0));
        
        // Spawn threads that interact with shared state
        let t1 = thread::spawn(move || {
            // Thread 1 operations
        });
        
        let t2 = thread::spawn(move || {
            // Thread 2 operations
        });
        
        // Join threads and verify correctness
        t1.join().unwrap();
        t2.join().unwrap();
        
        // Assert invariants hold
        assert_eq!(shared.load(Ordering::SeqCst), expected_value);
    });
}
```

## Continuous Integration

Loom tests are integrated into the CI pipeline:
- Run as a separate matrix entry in the `rust-coverage` job
- Execute in release mode for optimal performance (~16 seconds total)
- Coverage is collected and reported to Coveralls
- Run on every PR and push to develop branch

The CI configuration:
- Uses `RUSTFLAGS="--cfg loom"` to enable loom tests
- Runs with `cargo +nightly test --release -p vortex-scan --test loom_concurrency`
- Has a dedicated `loom` suite in the test matrix
- Timeout set to 120 minutes (though tests typically complete in under 20 seconds)

## References

- [Loom Documentation](https://docs.rs/loom/)
- [Loom GitHub Repository](https://github.com/tokio-rs/loom)
- [Concurrency Testing Best Practices](https://github.com/tokio-rs/loom#best-practices)