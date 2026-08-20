// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Matches over the arithmetic ptype of a [`SequenceData`](crate::SequenceData), which is always
/// `i64` or `u64`. Two arms rather than eight, which matters because the arithmetic ptype is
/// usually matched alongside the output ptype.
macro_rules! match_each_calculation_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use vortex_array::dtype::PType;
        match $self {
            PType::I64 => {
                type $enc = i64;
                $body
            }
            PType::U64 => {
                type $enc = u64;
                $body
            }
            other => vortex_error::vortex_panic!(
                "sequence arithmetic ptype must be i64 or u64, got {other}"
            ),
        }
    }};
}

pub(crate) use match_each_calculation_ptype;
