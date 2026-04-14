// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::dtype::PType;

/// Variant enum for Vortex primitive types.
#[non_exhaustive]
#[repr(C)]
#[expect(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum vx_ptype {
    /// Unsigned 8-bit integer
    PTYPE_U8 = 0,
    /// Unsigned 16-bit integer
    PTYPE_U16 = 1,
    /// Unsigned 32-bit integer
    PTYPE_U32 = 2,
    /// Unsigned 64-bit integer
    PTYPE_U64 = 3,
    /// Signed 8-bit integer
    PTYPE_I8 = 4,
    /// Signed 16-bit integer
    PTYPE_I16 = 5,
    /// Signed 32-bit integer
    PTYPE_I32 = 6,
    /// Signed 64-bit integer
    PTYPE_I64 = 7,
    /// 16-bit floating point number
    PTYPE_F16 = 8,
    /// 32-bit floating point number
    PTYPE_F32 = 9,
    /// 64-bit floating point number
    PTYPE_F64 = 10,
}

impl From<vx_ptype> for PType {
    fn from(value: vx_ptype) -> Self {
        match value {
            vx_ptype::PTYPE_U8 => PType::U8,
            vx_ptype::PTYPE_U16 => PType::U16,
            vx_ptype::PTYPE_U32 => PType::U32,
            vx_ptype::PTYPE_U64 => PType::U64,
            vx_ptype::PTYPE_I8 => PType::I8,
            vx_ptype::PTYPE_I16 => PType::I16,
            vx_ptype::PTYPE_I32 => PType::I32,
            vx_ptype::PTYPE_I64 => PType::I64,
            vx_ptype::PTYPE_F16 => PType::F16,
            vx_ptype::PTYPE_F32 => PType::F32,
            vx_ptype::PTYPE_F64 => PType::F64,
        }
    }
}

impl From<PType> for vx_ptype {
    fn from(value: PType) -> Self {
        match value {
            PType::U8 => vx_ptype::PTYPE_U8,
            PType::U16 => vx_ptype::PTYPE_U16,
            PType::U32 => vx_ptype::PTYPE_U32,
            PType::U64 => vx_ptype::PTYPE_U64,
            PType::I8 => vx_ptype::PTYPE_I8,
            PType::I16 => vx_ptype::PTYPE_I16,
            PType::I32 => vx_ptype::PTYPE_I32,
            PType::I64 => vx_ptype::PTYPE_I64,
            PType::F16 => vx_ptype::PTYPE_F16,
            PType::F32 => vx_ptype::PTYPE_F32,
            PType::F64 => vx_ptype::PTYPE_F64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vx_ptype_to_ptype() {
        assert_eq!(PType::from(vx_ptype::PTYPE_U8), PType::U8);
        assert_eq!(PType::from(vx_ptype::PTYPE_U16), PType::U16);
        assert_eq!(PType::from(vx_ptype::PTYPE_U32), PType::U32);
        assert_eq!(PType::from(vx_ptype::PTYPE_U64), PType::U64);
        assert_eq!(PType::from(vx_ptype::PTYPE_I8), PType::I8);
        assert_eq!(PType::from(vx_ptype::PTYPE_I16), PType::I16);
        assert_eq!(PType::from(vx_ptype::PTYPE_I32), PType::I32);
        assert_eq!(PType::from(vx_ptype::PTYPE_I64), PType::I64);
        assert_eq!(PType::from(vx_ptype::PTYPE_F16), PType::F16);
        assert_eq!(PType::from(vx_ptype::PTYPE_F32), PType::F32);
        assert_eq!(PType::from(vx_ptype::PTYPE_F64), PType::F64);
    }

    #[test]
    fn test_ptype_to_vx_ptype() {
        assert_eq!(vx_ptype::from(PType::U8), vx_ptype::PTYPE_U8);
        assert_eq!(vx_ptype::from(PType::U16), vx_ptype::PTYPE_U16);
        assert_eq!(vx_ptype::from(PType::U32), vx_ptype::PTYPE_U32);
        assert_eq!(vx_ptype::from(PType::U64), vx_ptype::PTYPE_U64);
        assert_eq!(vx_ptype::from(PType::I8), vx_ptype::PTYPE_I8);
        assert_eq!(vx_ptype::from(PType::I16), vx_ptype::PTYPE_I16);
        assert_eq!(vx_ptype::from(PType::I32), vx_ptype::PTYPE_I32);
        assert_eq!(vx_ptype::from(PType::I64), vx_ptype::PTYPE_I64);
        assert_eq!(vx_ptype::from(PType::F16), vx_ptype::PTYPE_F16);
        assert_eq!(vx_ptype::from(PType::F32), vx_ptype::PTYPE_F32);
        assert_eq!(vx_ptype::from(PType::F64), vx_ptype::PTYPE_F64);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let ptypes = vec![
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::F16,
            PType::F32,
            PType::F64,
        ];

        for ptype in ptypes {
            let vx = vx_ptype::from(ptype);
            let back = PType::from(vx);
            assert_eq!(ptype, back);
        }
    }

    #[test]
    fn test_vx_ptype_enum_values() {
        // Important: These values are part of the FFI ABI contract and must not change
        // as they may be used in serialization or by C clients
        assert_eq!(vx_ptype::PTYPE_U8 as i32, 0);
        assert_eq!(vx_ptype::PTYPE_U16 as i32, 1);
        assert_eq!(vx_ptype::PTYPE_U32 as i32, 2);
        assert_eq!(vx_ptype::PTYPE_U64 as i32, 3);
        assert_eq!(vx_ptype::PTYPE_I8 as i32, 4);
        assert_eq!(vx_ptype::PTYPE_I16 as i32, 5);
        assert_eq!(vx_ptype::PTYPE_I32 as i32, 6);
        assert_eq!(vx_ptype::PTYPE_I64 as i32, 7);
        assert_eq!(vx_ptype::PTYPE_F16 as i32, 8);
        assert_eq!(vx_ptype::PTYPE_F32 as i32, 9);
        assert_eq!(vx_ptype::PTYPE_F64 as i32, 10);
    }
}
