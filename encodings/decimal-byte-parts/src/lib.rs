// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod decimal_byte_parts;

use decimal_byte_parts::compute::is_constant::DecimalBytePartsIsConstantKernel;
/// This encoding allow compression of decimals using integer compression schemes.
/// Decimals can be compressed by narrowing the signed decimal value into the smallest signed value,
/// then integer compression if that is a value `ptype`, otherwise the decimal can be split into
/// parts.
/// These parts can be individually compressed.
/// This encoding will compress large signed decimals by removing the leading zeroes (after the sign)
/// an i128 decimal could be converted into a [i64, u64] with further narrowing applied to either
/// value.
pub use decimal_byte_parts::*;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize decimal-byte-parts encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(DecimalByteParts);

    session.aggregate_fns().register_aggregate_kernel(
        DecimalByteParts.id(),
        Some(IsConstant.id()),
        &DecimalBytePartsIsConstantKernel,
    );
}
