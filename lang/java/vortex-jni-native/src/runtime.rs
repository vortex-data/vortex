// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI entry points for tuning the shared [`CurrentThreadWorkerPool`](crate::POOL).

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::sys::jint;
use vortex::error::VortexExpect;

use crate::POOL;
use crate::errors::try_or_throw;

/// Set the number of background worker threads driving the JNI executor. Passing `0`
/// disables background execution — work is only driven when a Java thread calls a
/// blocking API. Passing a value larger than the current count spawns additional
/// workers; passing a smaller value signals excess workers to shut down.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeRuntime_setWorkerThreads(
    mut env: EnvUnowned,
    _class: JClass,
    n: jint,
) {
    try_or_throw(&mut env, |_| {
        if n < 0 {
            throw_runtime!("worker thread count must be non-negative");
        }
        POOL.set_workers(n as usize);
        Ok(())
    });
}

/// Set the number of background worker threads to `available_parallelism() - 1`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeRuntime_setWorkerThreadsToAvailableParallelism(
    _env: EnvUnowned,
    _class: JClass,
) {
    POOL.set_workers_to_available_parallelism();
}

/// Return the current number of background worker threads.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeRuntime_workerCount(
    _env: EnvUnowned,
    _class: JClass,
) -> jint {
    jint::try_from(POOL.worker_count()).vortex_expect("Must be able to convert to jint")
}
