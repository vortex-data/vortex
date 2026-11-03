// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use sqllogictest::ColumnType;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DFColumnType {
    Boolean,
    DateTime,
    Integer,
    Float,
    Text,
    Timestamp,
    Another,
}

impl ColumnType for DFColumnType {
    fn from_char(value: char) -> Option<Self> {
        Some(match value {
            'B' => Self::Boolean,
            'D' => Self::DateTime,
            'I' => Self::Integer,
            'P' => Self::Timestamp,
            'R' => Self::Float,
            'T' => Self::Text,
            _ => Self::Another,
        })
    }

    fn to_char(&self) -> char {
        match self {
            Self::Boolean => 'B',
            Self::DateTime => 'D',
            Self::Integer => 'I',
            Self::Timestamp => 'P',
            Self::Float => 'R',
            Self::Text => 'T',
            Self::Another => '?',
        }
    }
}
