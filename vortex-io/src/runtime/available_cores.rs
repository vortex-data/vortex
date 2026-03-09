use hwlocality::Topology;
use hwlocality::errors::RawHwlocError;
use hwlocality::object::types::ObjectType;

/// The number of available physical and logical cores.
///
/// "Available" means, for example, the fraction of cores available to this process's Linux cgroup.
///
/// "Physical core" refers to a core with its own ALUs and control unit.
///
/// "Logical core" refers to a [simultaneous multithreading
/// (SMT)](https://en.wikipedia.org/wiki/Simultaneous_multithreading) thread. An SMT thread
/// typically has its own registers but shares an ALU, MMU, control unit, and all its caches with
/// one or more other SMT threads.
///
/// CPU-bound work (like decompressing a Vortex array) is typically faster when executed
/// sequentially than when executed concurrently on two SMT threads which share a physical core.
#[derive(Debug)]
pub struct AvailableCores {
    /// The logical cores available to this process.
    ///
    /// We generally do not overestimate this value.
    pub logical: usize,
    /// The physical cores available to this process.
    ///
    /// We may overestimate this if hwlocality cannot interrogate this machine's hardware.
    pub physical: usize,
}

pub fn available_cores() -> AvailableCores {
    match hwloc_available_cores() {
        Ok(available_cores) => return available_cores,
        Err(error) => tracing::debug!(?error, "hwloc failed"),
    }

    match std::thread::available_parallelism() {
        Ok(available_parallelism) => {
            return AvailableCores {
                logical: available_parallelism.into(),
                physical: available_parallelism.into(),
            };
        }
        Err(error) => tracing::debug!(?error, "available parallelism failed"),
    }

    let n_cpus = num_cpus::get();

    AvailableCores {
        logical: n_cpus,
        physical: n_cpus,
    }
}

/// Use hwloc to determine the cores available to this process.
fn hwloc_available_cores() -> Result<AvailableCores, RawHwlocError> {
    let topo = Topology::new()?;

    let allowed = topo.allowed_cpuset();

    let physical = topo
        .objects_inside_cpuset_with_type(allowed, ObjectType::Core)
        .count();

    let logical = topo
        .objects_inside_cpuset_with_type(allowed, ObjectType::PU)
        .count();

    tracing::debug!(logical, physical, "hwloc");

    Ok(AvailableCores { logical, physical })
}
