pub enum ExecutionMode {
    /// The CPU-bound layout evaluation tasks are driven by the poller of the
    /// returned [`vortex_array::stream::ArrayStream`].
    Inline,
    /// The CPU-bound layout evaluation tasks are driven by a provided thread pool.
    // TODO(ngates): feature-flag this dependency.
    RayonThreadPool(rayon::ThreadPool),
}
