// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-encoding synthetic fixtures.
//!
//! Each fixture produces data patterns designed to exercise a specific stable encoding.

mod alp;
mod alprd;
mod bitpacked;
mod bytebool;
mod constant;
mod datetimeparts;
mod decimal_byte_parts;
mod delta;
mod dict;
mod for_;
mod fsst;
mod pco;
mod rle;
mod runend;
mod sequence;
mod sparse;
mod zigzag;
mod zstd;

use crate::fixtures::FlatLayoutFixture;

pub(crate) const N: usize = 1024;

/// All per-encoding fixtures.
pub fn fixtures() -> Vec<Box<dyn FlatLayoutFixture>> {
    vec![
        Box::new(alp::AlpFixture),
        Box::new(alprd::AlprdFixture),
        Box::new(bitpacked::BitPackedFixture),
        Box::new(bytebool::ByteBoolFixture),
        Box::new(datetimeparts::DateTimePartsFixture),
        Box::new(decimal_byte_parts::DecimalBytePartsFixture),
        Box::new(delta::DeltaFixture),
        Box::new(dict::DictFixture),
        Box::new(fsst::FsstFixture),
        Box::new(for_::FoRFixture),
        Box::new(pco::PcoFixture),
        Box::new(rle::RleFixture),
        Box::new(runend::RunEndFixture),
        Box::new(sequence::SequenceFixture),
        Box::new(sparse::SparseFixture),
        Box::new(zstd::ZstdFixture),
        Box::new(zigzag::ZigZagFixture),
        Box::new(constant::ConstantFixture),
    ]
}
