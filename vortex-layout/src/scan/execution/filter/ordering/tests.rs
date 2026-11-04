// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use parking_lot::Mutex;

use super::AdaptiveOrder;
use super::MIN_OBSERVATIONS;
use super::Observation;
use super::OrderState;

fn ordering(conjuncts: usize) -> AdaptiveOrder {
    AdaptiveOrder(Mutex::new(OrderState::new(conjuncts, 17)))
}

fn report(order: &AdaptiveOrder, index: usize, cost: u128, output_rows: usize) {
    order.report(
        index,
        Observation {
            version: 0,
            input_rows: 1000,
            output_rows,
            compute_ns: cost * 1000,
            recheck: false,
        },
    );
}

#[test]
fn expensive_selective_predicate_does_not_displace_a_cheap_filter() {
    let order = ordering(2);
    for _ in 0..MIN_OBSERVATIONS {
        report(&order, 0, 2, 500);
        report(&order, 1, 100, 1);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
}

#[test]
fn waits_for_evidence_then_moves_the_cheaper_filter_first() {
    let order = ordering(2);
    for _ in 0..MIN_OBSERVATIONS {
        report(&order, 0, 100, 500);
    }
    for _ in 0..MIN_OBSERVATIONS - 1 {
        report(&order, 1, 2, 500);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
    report(&order, 1, 2, 500);
    assert_eq!(order.0.lock().preferred.as_slice(), &[1, 0]);
}

#[test]
fn recent_selectivity_can_replace_a_long_history() {
    let order = ordering(2);
    for _ in 0..1000 {
        report(&order, 0, 10, 1);
        report(&order, 1, 10, 900);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
    for _ in 0..16 {
        report(&order, 0, 10, 990);
        report(&order, 1, 10, 1);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[1, 0]);
}

#[test]
fn small_estimated_improvements_do_not_switch_order() {
    let order = ordering(2);
    for _ in 0..MIN_OBSERVATIONS {
        report(&order, 0, 10, 500);
        report(&order, 1, 9, 500);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
}

#[test]
fn changed_bound_requires_fresh_evidence_and_ignores_older_results() {
    let order = ordering(2);
    for _ in 0..MIN_OBSERVATIONS {
        report(&order, 0, 10, 1);
        report(&order, 1, 10, 500);
    }
    order.version_changed(0, 2);
    report(&order, 0, 1, 0);
    assert_eq!(order.0.lock().estimates[0].observations, 0);
    for sample in 0..MIN_OBSERVATIONS {
        order.report(
            0,
            Observation {
                version: 2,
                input_rows: 1000,
                output_rows: 1000,
                compute_ns: 10_000,
                recheck: false,
            },
        );
        if sample + 1 < MIN_OBSERVATIONS {
            assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
        }
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[1, 0]);
    order.version_changed(0, 1);
    assert_eq!(order.0.lock().estimates[0].version, 2);
}

#[test]
fn rechecks_do_not_train_the_discovery_order() {
    let order = ordering(2);
    for _ in 0..100 {
        order.report(
            0,
            Observation {
                version: 0,
                input_rows: 3,
                output_rows: 0,
                compute_ns: 1,
                recheck: true,
            },
        );
    }
    assert_eq!(order.0.lock().estimates[0].observations, 0);
}

#[test]
fn exploration_observes_a_predicate_hidden_by_short_circuiting() {
    let order = ordering(2);
    let mut explored = 0;
    for _ in 0..4096 {
        let chunk = order.begin_chunk();
        if chunk.exploration {
            explored += 1;
            assert_eq!(chunk.indices.as_slice(), &[1, 0]);
            report(&order, 1, 100, 1000);
        } else {
            assert_eq!(chunk.indices.as_slice(), &[0, 1]);
        }
        report(&order, 0, 1, 0);
    }
    assert!((64..192).contains(&explored));
    assert_eq!(order.0.lock().estimates[1].observations, explored);
    assert_eq!(order.0.lock().preferred.as_slice(), &[0, 1]);
}

#[test]
fn an_admitted_chunk_keeps_its_order_while_other_chunks_train() {
    let order = ordering(2);
    let chunk = order.begin_chunk();
    let original = chunk.indices.clone();
    for _ in 0..MIN_OBSERVATIONS {
        report(&order, 0, 100, 500);
        report(&order, 1, 2, 500);
    }
    assert_eq!(order.0.lock().preferred.as_slice(), &[1, 0]);
    assert_eq!(chunk.indices, original);
}

#[test]
fn empty_inputs_do_not_create_invalid_estimates() {
    let order = ordering(2);
    order.report(
        0,
        Observation {
            version: 0,
            input_rows: 0,
            output_rows: 0,
            compute_ns: 0,
            recheck: false,
        },
    );
    assert_eq!(order.0.lock().estimates[0].observations, 0);
}

#[test]
fn exploration_preserves_every_predicate_once() {
    for conjuncts in 0..8 {
        let order = ordering(conjuncts);
        for _ in 0..128 {
            let chunk = order.begin_chunk();
            let mut indices = chunk.indices;
            indices.sort_unstable();
            assert_eq!(indices.as_slice(), (0..conjuncts).collect::<Vec<_>>());
            if conjuncts < 2 {
                assert!(!chunk.exploration);
            }
        }
    }
}
