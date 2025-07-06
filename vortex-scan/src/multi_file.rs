use std::path::{Path, PathBuf};
use std::thread;

use vortex_file::VortexFile;

use crate::iterator::MultiFileIterator;

pub struct MultiFileScan {
    file_paths: Vec<PathBuf>,
    vortex_files: Vec<VortexFile>,
    num_threads: usize,
}

impl MultiFileScan {
    pub fn new() -> Self {
        let num_threads = thread::available_parallelism()
            .map(|p| p.get())
            .expect("Failed to get number of available threads");

        Self {
            file_paths: Vec::new(),
            vortex_files: Vec::new(),
            num_threads,
        }
    }

    pub fn with_vortex_files<I>(mut self, vortex_files: I) -> Self
    where
        I: IntoIterator<Item = VortexFile>,
    {
        for vortex_file in vortex_files.into_iter() {
            self.vortex_files.push(vortex_file);
        }

        self
    }

    pub fn with_file_paths<P, I>(mut self, file_paths: I) -> Self
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        self.file_paths = file_paths
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        self
    }
}

impl MultiFileScan {
    pub fn into_array_iterator(self) -> MultiFileIterator {
        MultiFileIterator::new(self.num_threads)
            .with_file_paths(self.file_paths.clone())
            .with_vortex_files(self.vortex_files.clone())
    }
}

#[cfg(test)]
mod tests {
    use core::panic;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_distribute_files_to_threads() {
        let path1 = PathBuf::from(
            "/Users/lx/Code/vortex/bench-vortex/data/clickbench_partitioned/vortex-file-compressed/hits_0.vortex",
        );
        let path2 = PathBuf::from(
            "/Users/lx/Code/vortex/bench-vortex/data/clickbench_partitioned/vortex-file-compressed/hits_1.vortex",
        );

        let multi_file_iter = MultiFileScan::new()
            .with_file_paths([path1, path2])
            .into_array_iterator();
        let shared_iter = Arc::new(multi_file_iter);

        let mut thread_handles = vec![];

        for thread_id in 0..2 {
            let iter_clone = Arc::clone(&shared_iter);
            let handle = thread::spawn(move || {
                loop {
                    let result = iter_clone.next(thread_id);

                    match result {
                        Some(Ok(array_result)) => {
                            println!("Thread {}: {:?} bytes", thread_id, array_result.nbytes());
                        }
                        Some(Err(err)) => {
                            panic!("Thread {}: Error: {}", thread_id, err);
                        }
                        None => {
                            println!("Thread {}: No more results", thread_id);
                            break;
                        }
                    }
                }
            });
            thread_handles.push(handle);
        }

        for handle in thread_handles {
            handle.join().unwrap();
        }
    }
}
