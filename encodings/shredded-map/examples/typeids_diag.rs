// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_shredded_map::ShreddedMapArraySlotsExt;

#[path = "../benches/common/mod.rs"]
mod common;

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let idx = data.shredded.data().columns().iter().position(|c| &*c.key == "device").unwrap();
    let union = data.shredded.columns()[idx].clone().execute::<UnionArray>(&mut ctx).unwrap();
    let ids = union.type_ids().clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    let mask = ids.validity().unwrap().execute_mask(ids.len(), &mut ctx).unwrap();
    let mut counts = [0usize; 256];
    let mut runs = 0;
    let mut prev = None;
    for (i, &t) in ids.as_slice::<u8>().iter().enumerate() {
        let v = mask.value(i).then_some(t);
        if v != prev {
            runs += 1;
            prev = v;
        }
        if let Some(t) = v {
            counts[t as usize] += 1;
        }
    }
    println!("len {} valid {} runs {runs} counts {:?}", ids.len(), mask.true_count(), &counts[..4]);
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let c = common::compress(&ids.clone().into_array());
    println!("compressed type_ids: {} -> {} ({})", ids.nbytes(), c.nbytes(), c.encoding_id());
    let fresh = PrimitiveArray::new(
        vortex_buffer::Buffer::<u8>::copy_from(ids.as_slice::<u8>()),
        vortex_array::validity::Validity::from_mask(mask.clone(), vortex_array::dtype::Nullability::Nullable),
    )
    .into_array();
    let c = common::compress(&fresh);
    println!("fresh rebuild: -> {} ({})", c.nbytes(), c.encoding_id());
    let mut zeroed: Vec<u8> = ids.as_slice::<u8>().to_vec();
    let mut distinct_under_nulls = std::collections::BTreeMap::new();
    for (i, v) in zeroed.iter_mut().enumerate() {
        if !mask.value(i) {
            *distinct_under_nulls.entry(*v).or_insert(0usize) += 1;
            *v = 0;
        }
    }
    println!("values under nulls: {distinct_under_nulls:?}");
    let z = PrimitiveArray::new(
        vortex_buffer::Buffer::<u8>::from(zeroed),
        vortex_array::validity::Validity::from_mask(mask.clone(), vortex_array::dtype::Nullability::Nullable),
    )
    .into_array();
    let c = common::compress(&z);
    println!("nulls zeroed: -> {} ({})", c.nbytes(), c.encoding_id());
    println!("validity child: {:?}", ids.validity().unwrap());
    println!("stats: {:?}", ids.statistics().to_owned());
    // Synthetic: 1M u8 with long runs, null in a long run in the middle.
    let n = 1_000_000usize;
    for (label, null_frac) in [("all valid", 0.0), ("62% null in runs", 0.62), ("1% null", 0.01)] {
        let vals: Vec<Option<u8>> = (0..n)
            .map(|i| {
                let null = ((i / 2000) % 100) < (null_frac * 100.0) as usize;
                (!null).then_some(((i / 5000) % 2) as u8)
            })
            .collect();
        let arr = PrimitiveArray::from_option_iter(vals).into_array();
        let c = common::compress(&arr);
        println!("synthetic nullable u8 {label}: {} -> {} ({})", arr.nbytes(), c.nbytes(), c.encoding_id());
    }
    let raw: Vec<u8> = ids.as_slice::<u8>().to_vec();
    let c = common::compress(&PrimitiveArray::from_iter(raw).into_array());
    println!("same values non-null: -> {} ({})", c.nbytes(), c.encoding_id());
}
