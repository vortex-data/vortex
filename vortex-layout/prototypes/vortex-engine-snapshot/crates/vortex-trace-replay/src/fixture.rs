//! Synthetic-fixture generation.
//!
//! While the recorder (sub-project (1)) is not yet implemented, the
//! replay engine and viewer need a working `*.vtrx` to exercise.
//! This module produces a deterministic synthetic trace covering
//! every event variant at small scale.
//!
//! `vtrace gen-fixture <path>` writes a fixture file the viewer can
//! load.

use std::io::Write;

use vortex_trace_format::header::{
    BrokerInfo, ChannelInfo, ChannelTopology, OperatorInfo, ResourceInfo, TaskOptionsSnap,
    TraceHeader,
};
use vortex_trace_format::record::{
    AsyncId, BrokerId, ChannelId, InputPortId, InputPortRef, LatencyClass, OperatorId, PhaseKind,
    ProposalCostSnap, ProposalValueSnap, StepKind, TracePayload, TraceRecord, TurnOutcome,
    WorkClass, WorkerId,
};
use vortex_trace_format::serialized::{
    SerializedDomainSpan, SerializedRequirementSet, SerializedRequirementSpan,
};
use vortex_trace_format::snapshot::{
    AsyncSnap, BrokerSnapshot, ChannelSnapshot, HeapEntrySnap, InFlightRequest,
    PortRequirementSnap, TurnSnapshot, WorkerSnapshot,
};

use crate::writer::{TraceWriter, write_event, write_snapshot};

const NUM_TURNS: u32 = 12;
const NUM_WORKERS: u32 = 4;

pub fn write_fixture<W: Write>(mut out: W) -> std::io::Result<()> {
    let header = synthetic_header();
    let mut w = TraceWriter::new(&mut out, &header)?;

    let total_rows: u64 = 4096;
    let mut requirement_rows: u64 = 1024;
    let mut buffered_per_channel: Vec<u64> = vec![0; header.channels.len()];
    let mut memory_used: u64 = 0;

    for turn in 0..NUM_TURNS {
        // TurnBegin (main thread)
        write_event(
            w.inner_mut(),
            &TraceRecord {
                worker_id: WorkerId::MAIN,
                turn,
                payload: TracePayload::TurnBegin,
            },
        )?;

        // Phase: MaintainAsync (main)
        emit_phase(w.inner_mut(), turn, PhaseKind::MaintainAsync, |inner| {
            if turn == 1 {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::AsyncSubmitted {
                            op: OperatorId(2),
                            lane: 0,
                            async_id: AsyncId(100),
                            label: "scan-page-3".to_string(),
                            span: SerializedDomainSpan { start: 0, end: 512 },
                            latency_class: LatencyClass::Long,
                        },
                    },
                )?;
            }
            if turn == 4 {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::AsyncWake {
                            async_id: AsyncId(100),
                            label: "scan-page-3".to_string(),
                            span: SerializedDomainSpan { start: 0, end: 512 },
                        },
                    },
                )?;
            }
            Ok(())
        })?;

        // Phase: Propagate (main)
        emit_phase(w.inner_mut(), turn, PhaseKind::Propagate, |inner| {
            if turn == 0 {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::RequirementRootRequired {
                            rows: total_rows,
                        },
                    },
                )?;
            }
            requirement_rows = (requirement_rows + 256).min(total_rows);
            // Consumer input ports in the new graph:
            //   (filter, 0), (project, 0), (reduce, 0), (reduce, 1),
            //   (sink, 0)
            let consumer_ports: [(u32, u32); 5] = [(2, 0), (3, 0), (4, 0), (4, 1), (5, 0)];
            for (op_id, port) in consumer_ports {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::RequirementChanged {
                            port: InputPortRef {
                                op: OperatorId(op_id),
                                port: InputPortId(port),
                            },
                            requirement: SerializedRequirementSet::new(vec![
                                SerializedRequirementSpan {
                                    start: 0,
                                    end: requirement_rows,
                                    demand: 1,
                                },
                            ]),
                        },
                    },
                )?;
            }
            Ok(())
        })?;

        // Phase: AdmitBrokers (main)
        emit_phase(w.inner_mut(), turn, PhaseKind::AdmitBrokers, |inner| {
            if turn % 3 == 1 {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::BrokerProposalEnqueued {
                            broker: BrokerId(0),
                            score: 0.85 - (turn as f32) * 0.01,
                            value: ProposalValueSnap {
                                rows: 256,
                                bytes: 16_384,
                            },
                            cost: ProposalCostSnap {
                                class: WorkClass::Io,
                                estimated_micros: 600,
                            },
                        },
                    },
                )?;
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::BrokerSubmit {
                            broker: BrokerId(0),
                            label: "fetch-batch".to_string(),
                            latency: LatencyClass::Long,
                            required_rows: 256,
                            score: 0.85 - (turn as f32) * 0.01,
                        },
                    },
                )?;
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::BrokerPull {
                            broker: BrokerId(0),
                            request_id: turn as u64,
                            count: 256,
                        },
                    },
                )?;
            }
            if turn % 3 == 2 && turn > 1 {
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::SubstrateComplete {
                            broker: BrokerId(0),
                            request_id: (turn - 1) as u64,
                            result_summary: "256 rows".to_string(),
                        },
                    },
                )?;
            }
            Ok(())
        })?;

        // Phase: WorkerSteps — each worker pops + runs
        // Map worker -> a (op, output_channel_id) pair so each push
        // lands on the right channel for the new 6-op graph.
        // worker 0 -> filter   (out: ch1, SPMC)
        // worker 1 -> project  (out: ch2, MPSC)
        // worker 2 -> reduce   (out: ch3, MPMC)
        // worker 3 -> scan-l   (out: ch0, SPSC)
        let work_assignments: [(OperatorId, ChannelId); NUM_WORKERS as usize] = [
            (OperatorId(2), ChannelId(1)),
            (OperatorId(3), ChannelId(2)),
            (OperatorId(4), ChannelId(3)),
            (OperatorId(0), ChannelId(0)),
        ];
        emit_phase(w.inner_mut(), turn, PhaseKind::WorkerSteps, |inner| {
            for worker in 0..NUM_WORKERS {
                let (op, out_channel) = work_assignments[worker as usize];
                let lane = worker % 2;
                let score = 0.9 - (worker as f32) * 0.05 - (turn as f32) * 0.005;
                // Enqueue a fresh proposal
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::ProposalEnqueued {
                            op,
                            lane,
                            class: WorkClass::Cpu,
                            score,
                            value: ProposalValueSnap {
                                rows: 128,
                                bytes: 8_192,
                            },
                            cost: ProposalCostSnap {
                                class: WorkClass::Cpu,
                                estimated_micros: 50,
                            },
                        },
                    },
                )?;
                // Pop + step
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::ProposalPopped { op, lane, score },
                    },
                )?;
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::WorkerStepBegin {
                            op,
                            lane,
                            kind: StepKind::Run,
                        },
                    },
                )?;
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::OperatorRun {
                            op,
                            lane,
                            label: format!("op#{}/{} run", op.0, lane),
                            class: WorkClass::Cpu,
                            score,
                        },
                    },
                )?;
                // Push some bytes onto the operator's output channel.
                let ch_idx = out_channel.0 as usize;
                let push_bytes = 1024;
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::ChannelPush {
                            channel: out_channel,
                            op,
                            lane,
                            span: SerializedDomainSpan {
                                start: (turn as u64) * 128,
                                end: (turn as u64) * 128 + 128,
                            },
                            rows: 128,
                            bytes: push_bytes,
                        },
                    },
                )?;
                if ch_idx < buffered_per_channel.len() {
                    buffered_per_channel[ch_idx] =
                        buffered_per_channel[ch_idx].saturating_add(push_bytes);
                }
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId(worker),
                        turn,
                        payload: TracePayload::WorkerStepEnd { op, lane },
                    },
                )?;
            }
            Ok(())
        })?;

        // Phase: Classify (main) — memory grant adjustment + maybe seal
        emit_phase(w.inner_mut(), turn, PhaseKind::Classify, |inner| {
            memory_used = memory_used.saturating_add(2048);
            if turn % 2 == 0 {
                let mut changes = Vec::new();
                for ch in &header.channels {
                    changes.push(vortex_trace_format::record::ChannelGrantChange {
                        channel: ch.id,
                        old_capacity_bytes: ch.initial_capacity_bytes,
                        new_capacity_bytes: ch.initial_capacity_bytes
                            + (turn as u64) * 1024,
                    });
                }
                write_event(
                    inner,
                    &TraceRecord {
                        worker_id: WorkerId::MAIN,
                        turn,
                        payload: TracePayload::MemoryGrantChanged { changes },
                    },
                )?;
            }
            if turn == NUM_TURNS - 1 {
                for ch in &header.channels {
                    write_event(
                        inner,
                        &TraceRecord {
                            worker_id: WorkerId::MAIN,
                            turn,
                            payload: TracePayload::ChannelSeal { channel: ch.id },
                        },
                    )?;
                }
            }
            Ok(())
        })?;

        // TurnEnd
        let outcome = if turn == NUM_TURNS - 1 {
            TurnOutcome::Done
        } else {
            TurnOutcome::Progress
        };
        write_event(
            w.inner_mut(),
            &TraceRecord {
                worker_id: WorkerId::MAIN,
                turn,
                payload: TracePayload::TurnEnd { outcome },
            },
        )?;

        // Snapshot at turn end
        let snap = build_snapshot(turn, &header, &buffered_per_channel, requirement_rows, memory_used);
        write_snapshot(w.inner_mut(), &snap)?;
    }

    Ok(())
}

fn emit_phase<W: Write, F>(
    out: &mut W,
    turn: u32,
    phase: PhaseKind,
    body: F,
) -> std::io::Result<()>
where
    F: FnOnce(&mut W) -> std::io::Result<()>,
{
    write_event(
        out,
        &TraceRecord {
            worker_id: WorkerId::MAIN,
            turn,
            payload: TracePayload::PhaseBegin { phase },
        },
    )?;
    body(out)?;
    write_event(
        out,
        &TraceRecord {
            worker_id: WorkerId::MAIN,
            turn,
            payload: TracePayload::PhaseEnd { phase },
        },
    )?;
    Ok(())
}

fn synthetic_header() -> TraceHeader {
    TraceHeader {
        format_version: 1,
        recorder_version: "fixture-0.1".to_string(),
        task_options: TaskOptionsSnap {
            max_turns: NUM_TURNS as u64,
            memory_limit_bytes: 64 * 1024 * 1024,
            worker_count: NUM_WORKERS,
        },
        operators: vec![
            OperatorInfo {
                id: OperatorId(0),
                name: "scan(lineitem)".to_string(),
                kind: "TableScan".to_string(),
                input_ports: vec![],
                output_ports: vec!["out".to_string()],
                lane_count: NUM_WORKERS,
            },
            OperatorInfo {
                id: OperatorId(1),
                name: "scan(orders)".to_string(),
                kind: "TableScan".to_string(),
                input_ports: vec![],
                output_ports: vec!["out".to_string()],
                lane_count: 2,
            },
            OperatorInfo {
                id: OperatorId(2),
                name: "filter(quantity > 10)".to_string(),
                kind: "MaskFilter".to_string(),
                input_ports: vec!["in".to_string()],
                output_ports: vec!["out".to_string()],
                lane_count: NUM_WORKERS,
            },
            OperatorInfo {
                id: OperatorId(3),
                name: "project(*, price * 1.05)".to_string(),
                kind: "Projection".to_string(),
                input_ports: vec!["in".to_string()],
                output_ports: vec!["out".to_string()],
                lane_count: 2,
            },
            OperatorInfo {
                id: OperatorId(4),
                name: "reduce(sum)".to_string(),
                kind: "GroupedAggregate".to_string(),
                input_ports: vec!["primary".to_string(), "side".to_string()],
                output_ports: vec!["out".to_string()],
                lane_count: 2,
            },
            OperatorInfo {
                id: OperatorId(5),
                name: "sink".to_string(),
                kind: "Sink".to_string(),
                input_ports: vec!["in".to_string()],
                output_ports: vec![],
                lane_count: 2,
            },
        ],
        channels: vec![
            // SPSC: scan(lineitem) -> filter
            ChannelInfo {
                id: ChannelId(0),
                name: "scan-l→filter".to_string(),
                topology: ChannelTopology::Spsc,
                producers: vec![OperatorId(0)],
                consumers: vec![InputPortRef {
                    op: OperatorId(2),
                    port: InputPortId(0),
                }],
                initial_capacity_bytes: 128 * 1024,
            },
            // SPMC: filter -> project, reduce(primary)
            ChannelInfo {
                id: ChannelId(1),
                name: "filter→{project, reduce.primary}".to_string(),
                topology: ChannelTopology::Spmc,
                producers: vec![OperatorId(2)],
                consumers: vec![
                    InputPortRef {
                        op: OperatorId(3),
                        port: InputPortId(0),
                    },
                    InputPortRef {
                        op: OperatorId(4),
                        port: InputPortId(0),
                    },
                ],
                initial_capacity_bytes: 128 * 1024,
            },
            // MPSC: scan(orders) + project -> reduce(side)
            ChannelInfo {
                id: ChannelId(2),
                name: "{scan-o, project}→reduce.side".to_string(),
                topology: ChannelTopology::Mpsc,
                producers: vec![OperatorId(1), OperatorId(3)],
                consumers: vec![InputPortRef {
                    op: OperatorId(4),
                    port: InputPortId(1),
                }],
                initial_capacity_bytes: 96 * 1024,
            },
            // MPMC: reduce -> sink (multi-lane work-stealing)
            ChannelInfo {
                id: ChannelId(3),
                name: "reduce→sink".to_string(),
                topology: ChannelTopology::Mpmc,
                producers: vec![OperatorId(4)],
                consumers: vec![InputPortRef {
                    op: OperatorId(5),
                    port: InputPortId(0),
                }],
                initial_capacity_bytes: 64 * 1024,
            },
        ],
        brokers: vec![BrokerInfo {
            id: BrokerId(0),
            name: "object-store".to_string(),
            label: "S3 reader".to_string(),
        }],
        resources: vec![ResourceInfo {
            id: 0,
            name: "stats(lineitem)".to_string(),
            producer: OperatorId(0),
        }],
        recorded_at_unix_secs: 1_715_212_800,
    }
}

fn build_snapshot(
    turn: u32,
    header: &TraceHeader,
    buffered: &[u64],
    requirement_rows: u64,
    memory_used: u64,
) -> TurnSnapshot {
    // Same op assignment as work_assignments above so snapshots
    // line up with the events.
    let work_ops: [u32; NUM_WORKERS as usize] = [2, 3, 4, 0];
    let workers = (0..NUM_WORKERS)
        .map(|w| WorkerSnapshot {
            worker_id: WorkerId(w),
            heap: vec![HeapEntrySnap {
                op: OperatorId(work_ops[w as usize]),
                lane: w % 2,
                class: WorkClass::Cpu,
                score: 0.7 - (w as f32) * 0.03,
                value: ProposalValueSnap {
                    rows: 64,
                    bytes: 4_096,
                },
                cost: ProposalCostSnap {
                    class: WorkClass::Cpu,
                    estimated_micros: 25,
                },
            }],
        })
        .collect();
    let channels = header
        .channels
        .iter()
        .enumerate()
        .map(|(i, c)| ChannelSnapshot {
            channel: c.id,
            buffered: vec![SerializedDomainSpan { start: 0, end: 128 }],
            buffered_bytes: buffered.get(i).copied().unwrap_or(0).min(c.initial_capacity_bytes),
            capacity_bytes: c.initial_capacity_bytes + (turn as u64) * 1024,
            output_requirement: SerializedRequirementSet::new(vec![
                SerializedRequirementSpan {
                    start: 0,
                    end: requirement_rows,
                    demand: 1,
                },
            ]),
        })
        .collect();
    let brokers = header
        .brokers
        .iter()
        .map(|b| BrokerSnapshot {
            broker: b.id,
            in_flight: if turn % 3 == 1 || turn % 3 == 2 {
                vec![InFlightRequest {
                    request_id: turn as u64,
                    label: "fetch-batch".to_string(),
                    since_turn: turn,
                }]
            } else {
                vec![]
            },
            pending_proposals: vec![],
        })
        .collect();
    let async_in_flight = if turn >= 1 && turn < 4 {
        vec![AsyncSnap {
            async_id: AsyncId(100),
            label: "scan-page-3".to_string(),
            span: SerializedDomainSpan { start: 0, end: 512 },
            since_turn: 1,
            latency_class: LatencyClass::Long,
        }]
    } else {
        vec![]
    };
    let consumer_ports: [(u32, u32); 5] = [(2, 0), (3, 0), (4, 0), (4, 1), (5, 0)];
    let requirements = consumer_ports
        .iter()
        .map(|&(op, port)| PortRequirementSnap {
            port: InputPortRef {
                op: OperatorId(op),
                port: InputPortId(port),
            },
            requirement: SerializedRequirementSet::new(vec![SerializedRequirementSpan {
                start: 0,
                end: requirement_rows,
                demand: 1,
            }]),
        })
        .collect();
    TurnSnapshot {
        turn,
        workers,
        channels,
        brokers,
        async_in_flight,
        requirements,
        lane_finished: vec![0; ((header.operators.len() as u32) * NUM_WORKERS) as usize],
        lane_owner: vec![0; ((header.operators.len() as u32) * NUM_WORKERS) as usize],
        memory_used_bytes: memory_used,
    }
}

