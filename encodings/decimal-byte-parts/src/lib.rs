// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An encoding that compresses decimals with integer compression schemes.
//!
//! A decimal is split into a signed most significant part (MSP) plus up to three `u64` lower
//! parts, and each part can be compressed independently. A decimal whose values fit an `i64`
//! needs no lower parts. Only the MSP narrows: it can use the smallest signed integer dtype that
//! holds its values, while every lower part stays `u64`. Decoding returns the canonical decimal
//! array, or a decimal [`Widen`] over a narrower child that can reuse a narrow MSP.
//!
//! [`Widen`]: vortex_array::arrays::Widen

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

use crate::decimal_byte_parts::compute::is_constant::DecimalBytePartsIsConstantKernel;

mod decimal_byte_parts;
pub use decimal_byte_parts::*;

/// Initialize decimal-byte-parts encoding in the given session.
pub fn initialize(session: &VortexSession) {
    // One plugin owns both serialized formats: registering it reads either ID and writes the
    // one that fits the array. Which of them a writer may emit is decided by its editions.
    session.arrays().register(DecimalBytePartsPlugin);
    compute::kernel::initialize(session);

    session.aggregate_fns().register_aggregate_kernel(
        DecimalByteParts.id(),
        Some(IsConstant.id()),
        &DecimalBytePartsIsConstantKernel,
    );
}
