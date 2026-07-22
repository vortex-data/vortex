// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::thread;

    use vortex::io::runtime::BlockingRuntime;
    use vortex_ffi::ffi_runtime;
    use vortex_ffi::vx_runtime_worker_count;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn runtime_is_host_thread_driven_by_default() {
        assert_eq!(vx_runtime_worker_count(), 0);

        let host_thread = thread::current().id();
        let task = ffi_runtime()
            .handle()
            .spawn(async move { thread::current().id() });
        let executor_thread = ffi_runtime().block_on(task);

        assert_eq!(executor_thread, host_thread);
    }
}
