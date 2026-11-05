// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reference-counting costs of [`ArrayRef`]: allocation, clone/drop on the allocating thread,
//! clone/drop on another thread, and handing arrays to another thread to drop.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::mpsc;
use std::thread;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const N: usize = 1024;

fn new_array() -> ArrayRef {
    PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array()
}

/// Allocate and drop `N` arrays on the same thread.
#[divan::bench]
fn alloc_drop(bencher: Bencher) {
    bencher.bench_local(|| {
        for _ in 0..N {
            drop(black_box(new_array()));
        }
    });
}

/// Clone and drop an array `N` times on the thread that allocated it.
#[divan::bench]
fn clone_drop_owner_thread(bencher: Bencher) {
    let array = new_array();
    bencher.bench_local(|| {
        for _ in 0..N {
            drop(black_box(array.clone()));
        }
    });
}

/// Clone and drop an array `N` times on a thread other than the one that allocated it.
#[divan::bench]
fn clone_drop_other_thread(bencher: Bencher) {
    let array = thread::spawn(new_array).join().unwrap();
    bencher.bench_local(|| {
        for _ in 0..N {
            drop(black_box(array.clone()));
        }
    });
}

/// Allocate `N` arrays on this thread and drop them on another one, as when a scan task
/// hands batches to a consumer.
#[divan::bench]
fn alloc_then_drop_on_other_thread(bencher: Bencher) {
    let (tx, rx) = mpsc::sync_channel::<Vec<ArrayRef>>(1);
    let (done_tx, done_rx) = mpsc::sync_channel::<()>(1);
    let consumer = thread::spawn(move || {
        while let Ok(batch) = rx.recv() {
            drop(batch);
            done_tx.send(()).unwrap();
        }
    });
    bencher.bench_local(|| {
        let batch: Vec<ArrayRef> = (0..N).map(|_| new_array()).collect();
        tx.send(batch).unwrap();
        done_rx.recv().unwrap();
    });
    drop(tx);
    consumer.join().unwrap();
}

/// Allocate `N` arrays on another thread and clone/drop each one here.
#[divan::bench]
fn receive_and_clone_drop(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            thread::spawn(|| (0..N).map(|_| new_array()).collect::<Vec<_>>())
                .join()
                .unwrap()
        })
        .bench_local_values(|batch| {
            for array in &batch {
                drop(black_box(array.clone()));
            }
            batch
        });
}
