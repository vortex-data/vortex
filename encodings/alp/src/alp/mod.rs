// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array support for classic ALP, on top of the `alp` crate's encoder.

pub use ::alp::ALPFloat;
pub use ::alp::Exponents;
pub use array::*;
pub use compress::alp_encode;
pub use decompress::decompress_into_array;
use vortex_session::VortexSession;

mod array;
mod compress;
pub(crate) mod compute;
mod decompress;
mod ops;
mod plugin;
mod rules;

pub(crate) use plugin::ALPPatchedPlugin;

pub(crate) fn initialize(session: &VortexSession) {
    rules::initialize(session);
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;

    use crate::alp::array::ALPMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            &ALPMetadata {
                patches: Some(PatchesMetadata::new(
                    usize::MAX,
                    usize::MAX,
                    PType::U64,
                    None,
                    None,
                    None,
                )),
                exp_e: u32::MAX,
                exp_f: u32::MAX,
            }
            .encode_to_vec(),
        );
    }
}
