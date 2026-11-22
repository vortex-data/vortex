// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks of the pipeline graph and of what each run reports, compiled into debug builds only.
//!
//! The scheduler trusts sources to report truthfully and the compiler to wire ports so no cycle
//! of full ports can form. A broken promise shows up much later as "Scan is stuck" or as a
//! wrong result; these checks fail at the pipeline that broke it.

use super::Core;
use super::PipelineId;
use super::Progress;
use super::SourceBlocked;
use super::SourceEnd;

impl Core {
    /// Checks the graph after a compile: ports and pipelines agree on their ends, every port has
    /// a reader, and fan-out is unbounded.
    pub(super) fn validate(&self) {
        for (id, pipe) in self.pipelines.iter().enumerate() {
            let Some(pipe) = pipe else { continue };
            for (index, &port) in pipe.inlets.iter().enumerate() {
                let source = self.arena.get(port).source;
                assert!(
                    source == (SourceEnd::Pipeline { id, inlet: index })
                        || self.arena.get(port).closed(),
                    "port {port} is an inlet of pipeline {id} but its source is {source:?}"
                );
            }
            for &port in &pipe.outlets {
                let queue = self.arena.get(port);
                assert!(
                    queue.sink == Some(id) || queue.closed(),
                    "port {port} is an outlet of pipeline {id} but its sink is {:?}",
                    queue.sink
                );
                // A pipeline blocked on one full outlet starves its other readers. If two of
                // them meet again downstream, the join waits on the starved one while the fork
                // waits on the join: a diamond deadlock. Unbounded fan-out never blocks.
                assert!(
                    pipe.outlets.len() == 1 || queue.unbounded(),
                    "pipeline {id} fans out into bounded port {port}, which can deadlock a \
                     diamond"
                );
            }
        }
    }

    /// Checks what one run of pipeline `id` reported: a source waits only on a condition some
    /// other event will clear.
    pub(super) fn validate_progress(&self, id: PipelineId, progress: &Progress) {
        let pipe = self.pipeline(id);
        match progress {
            Progress::Blocked(SourceBlocked::Inlet(index)) => {
                assert!(
                    *index < pipe.inlets.len(),
                    "pipeline {id} waits on inlet {index} of {}",
                    pipe.inlets.len()
                );
                let port = pipe.inlets[*index];
                let queue = self.arena.get(port);
                // Only the sink pushing or closing wakes the source, and it can do neither
                // into a closed or full port.
                assert!(
                    !queue.closed(),
                    "pipeline {id} waits on inlet {index} (port {port}), which is closed"
                );
                assert!(
                    queue.has_room(),
                    "pipeline {id} waits on inlet {index} (port {port}), which is full, so its \
                     sink {:?} can never push to wake it",
                    queue.sink
                );
            }
            Progress::Blocked(SourceBlocked::Io) => assert!(
                pipe.pending > 0 || !pipe.bytes.is_empty(),
                "pipeline {id} waits for a read with none in flight or delivered"
            ),
            Progress::Done => {
                for (index, &port) in pipe.inlets.iter().enumerate() {
                    assert!(
                        self.arena.get(port).closed(),
                        "pipeline {id} finished before inlet {index} (port {port}) closed, so \
                         its sink {:?} would push into a port nobody reads",
                        self.arena.get(port).sink
                    );
                }
            }
            Progress::Outlet => {}
        }
    }
}
