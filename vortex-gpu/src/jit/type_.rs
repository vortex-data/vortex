// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;

use vortex_dtype::PType;

pub struct CUDAType(&'static str);

impl Display for CUDAType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl From<PType> for CUDAType {
    fn from(value: PType) -> Self {
        CUDAType(match value {
            PType::U8 => "unsigned char",
            PType::U16 => "unsigned short",
            PType::U32 => "unsigned int",
            PType::U64 => "unsigned long long",
            PType::I8 => "char",
            PType::I16 => "short",
            PType::I32 => "int",
            PType::I64 => "long long",
            PType::F32 => "float",
            PType::F64 => "double",
            PType::F16 => todo!(),
        })
    }
}
