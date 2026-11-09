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
mod rules;

pub(crate) use array::ALPPatchedPlugin;

pub(crate) fn initialize(session: &VortexSession) {
    rules::initialize(session);
    rules::register_parent_rules(session);
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ALP;
    use crate::Exponents;
    use crate::alp::array::ALPMetadata;
    use crate::alp_encode;

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

    #[rstest]
    fn serde_roundtrip_sliced_chunked_patches(
        #[values(500, 1500)] start: usize,
    ) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let values =
            PrimitiveArray::from_iter((0..4096).map(|i| if i % 7 == 0 { PI } else { 1.0 }));
        let encoded = alp_encode(values.as_view(), Some(Exponents { e: 0, f: 0 }), &mut ctx)?;
        let sliced = <ALP as SliceKernel>::slice(encoded.as_view(), start..4096, &mut ctx)?
            .expect("ALP slices");
        let serialization =
            ArrayPlugin::serialize(&ALP, &sliced, &session)?.expect("ALP serializes");
        let read = ArrayPlugin::deserialize(
            &ALP,
            ArrayDeserialization::new(
                serialization.serialized_id,
                sliced.dtype(),
                sliced.len(),
                &serialization.metadata,
                &[],
                &serialization.children,
            ),
            &session,
        )?;
        let expected = values.into_array().slice(start..4096)?;

        let patch_index = ((start / 1024 + 1) * 1024).next_multiple_of(7) - start;
        assert_eq!(
            read.execute_scalar(patch_index, &mut ctx)?,
            expected.execute_scalar(patch_index, &mut ctx)?
        );
        assert_arrays_eq!(read, expected, &mut ctx);
        Ok(())
    }
}
