// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::ptr;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use flatbuffers::root;
use vortex_array::ArrayRef;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_error::VortexResult;
use vortex_file::EOF_SIZE;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::POSTSCRIPT_LEN_BYTES;
use vortex_file::VERSION_BYTES;
use vortex_file::flatbuffers::footer as fb;

use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::RUNTIME;
use crate::SESSION;
use crate::file::WriteSettings;
use crate::file::has_duplicate_field_names;
use crate::file::has_nullable_struct;

const MAX_CORRUPTIONS: usize = 8;

#[derive(Clone, Copy)]
pub enum SegmentField {
    /// Although valid Offset is u64, generated files are usually small, so most
    /// such offsets are rejected in file size check. Same for Length
    Offset(u16),
    Length(u16),
    /// Valid values are 0-16, we generate 17 as well
    AlignmentExponent(u8),
}

#[derive(Clone, Copy)]
pub enum Region {
    DataSegment(u8),
    Dtype,
    Layout,
    Statistics,
    Footer,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataSegment(index) => write!(f, "data_segment[{index}]"),
            Self::Dtype => f.write_str("dtype"),
            Self::Layout => f.write_str("layout"),
            Self::Statistics => f.write_str("statistics"),
            Self::Footer => f.write_str("footer"),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Corruption {
    FooterSegmentSpec {
        index: u8,
        field: SegmentField,
    },
    Bytes {
        region: Region,
        offset: u32,
        byte: u8,
    },
}

impl Corruption {
    fn is_noop(&self) -> bool {
        matches!(self, Self::Bytes { byte: 0, .. })
    }
}

impl fmt::Debug for Corruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FooterSegmentSpec { index, field } => {
                let (name, value) = match field {
                    SegmentField::Offset(v) => ("offset", u32::from(*v)),
                    SegmentField::Length(v) => ("length", u32::from(*v)),
                    SegmentField::AlignmentExponent(v) => ("alignment_exponent", u32::from(*v)),
                };
                write!(f, "footer.segments[{index}].{name} = {value}")
            }
            Self::Bytes {
                region,
                offset,
                byte,
            } => {
                write!(f, "{region:?}.bytes[{offset}] ^= {byte:#04x}")
            }
        }
    }
}

impl<'a> Arbitrary<'a> for Corruption {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0u8..=7)? {
            kind @ 0..=2 => Self::FooterSegmentSpec {
                index: u.arbitrary()?,
                field: match kind {
                    0 => SegmentField::Offset(u.arbitrary()?),
                    1 => SegmentField::Length(u.arbitrary()?),
                    _ => SegmentField::AlignmentExponent(u.int_in_range(0..=17)?),
                },
            },
            kind => Self::Bytes {
                region: match kind {
                    3 => Region::DataSegment(u.arbitrary()?),
                    4 => Region::Dtype,
                    5 => Region::Layout,
                    6 => Region::Statistics,
                    _ => Region::Footer,
                },
                offset: u.arbitrary()?,
                byte: u.arbitrary()?,
            },
        })
    }
}

const NO_CORRUPTION: Corruption = Corruption::Bytes {
    region: Region::Dtype,
    offset: 0,
    byte: 0,
};

pub struct FuzzFileCorruption {
    pub array: ArrayRef,
    pub corruptions: [Corruption; MAX_CORRUPTIONS],
    pub write_settings: WriteSettings,
}

impl fmt::Debug for FuzzFileCorruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let corruptions: Vec<_> = self.corruptions.iter().filter(|c| !c.is_noop()).collect();
        f.debug_struct("FuzzFileCorruption")
            .field("array", &self.array)
            .field(
                "corruptions (offset % len, segment index % segments)",
                &corruptions,
            )
            .field("write_settings", &self.write_settings)
            .finish()
    }
}

impl<'a> Arbitrary<'a> for FuzzFileCorruption {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let config = ArbitraryArrayConfig {
            dtype: None,
            len: 0..=FUZZ_FILE_ARRAY_MAX_LEN,
        };
        let array = ArbitraryArray::arbitrary_with_config(u, &config)?.0;
        let mut corruptions = [NO_CORRUPTION; MAX_CORRUPTIONS];
        let corruptions_len = u.int_in_range(0..=MAX_CORRUPTIONS)?;
        for slot in corruptions.iter_mut().take(corruptions_len) {
            *slot = u.arbitrary()?;
        }
        Ok(Self {
            array,
            corruptions,
            write_settings: WriteSettings::arbitrary(u)?,
        })
    }
}

pub fn run_file_corruption(fuzz: FuzzFileCorruption) -> VortexResult<bool> {
    let FuzzFileCorruption {
        array,
        corruptions,
        write_settings,
    } = fuzz;
    if has_nullable_struct(array.dtype()) || has_duplicate_field_names(array.dtype()) {
        return Ok(false);
    }

    let mut file = Vec::new();
    write_settings
        .write_options()
        .blocking(&*RUNTIME)
        .write(&mut file, array.to_array_iterator())?;

    for corruption in corruptions {
        match corruption {
            Corruption::FooterSegmentSpec { index, field } => {
                apply_footer_segment_spec(&mut file, index, field);
            }
            Corruption::Bytes {
                region,
                offset,
                byte,
            } => apply_bytes(&mut file, region, offset, byte),
        }
    }

    let Ok(vortex_file) = SESSION.open_options().open_buffer(file) else {
        return Ok(true);
    };
    let Ok(scan) = vortex_file.scan() else {
        return Ok(true);
    };
    let Ok(iter) = scan.into_array_iter(&*RUNTIME) else {
        return Ok(true);
    };
    let mut ctx = SESSION.create_execution_ctx();
    for batch in iter {
        let Ok(array) = batch else {
            return Ok(true);
        };
        let _canonical = array.execute::<RecursiveCanonical>(&mut ctx);
    }
    Ok(true)
}

fn postscript(file: &[u8]) -> Option<fb::Postscript<'_>> {
    let eof_start = file.len() - EOF_SIZE;
    let start = eof_start + VERSION_BYTES;
    let end = start + POSTSCRIPT_LEN_BYTES;
    let len_bytes: [u8; POSTSCRIPT_LEN_BYTES] = file.get(start..end)?.try_into().ok()?;
    let postscript_len = usize::from(u16::from_le_bytes(len_bytes));
    let start = eof_start.checked_sub(postscript_len)?;
    root::<fb::Postscript>(file.get(start..eof_start)?).ok()
}

fn span(file: &[u8], offset: u64, length: u32) -> Option<(usize, usize)> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = start.checked_add(length)?;
    (end <= file.len()).then_some((start, length))
}

fn segment_spec(file: &[u8], index: u8) -> Option<&fb::SegmentSpec> {
    let footer = postscript(file)?.footer()?;
    let (start, length) = span(file, footer.offset(), footer.length())?;
    let specs = root::<fb::Footer>(file.get(start..start + length)?)
        .ok()?
        .segment_specs()?;
    if specs.is_empty() {
        return None;
    }
    Some(specs.get(usize::from(index) % specs.len()))
}

fn patched_segment_spec(file: &[u8], index: u8, field: SegmentField) -> Option<(usize, [u8; 16])> {
    let spec = segment_spec(file, index)?;
    let position = (ptr::from_ref(spec) as usize).checked_sub(file.as_ptr() as usize)?;
    let mut patched = fb::SegmentSpec(spec.0);
    match field {
        SegmentField::Offset(value) => patched.set_offset(u64::from(value)),
        SegmentField::Length(value) => patched.set_length(u32::from(value)),
        SegmentField::AlignmentExponent(value) => patched.set_alignment_exponent(value),
    }
    Some((position, patched.0))
}

fn apply_footer_segment_spec(file: &mut [u8], index: u8, field: SegmentField) {
    let Some((position, bytes)) = patched_segment_spec(file, index, field) else {
        return;
    };
    if let Some(slot) = file.get_mut(position..position + bytes.len()) {
        slot.copy_from_slice(&bytes);
    }
}

fn region_span(file: &[u8], region: Region) -> Option<(usize, usize)> {
    let segment = match region {
        Region::DataSegment(index) => {
            let spec = segment_spec(file, index)?;
            return span(file, spec.offset(), spec.length());
        }
        Region::Dtype => postscript(file)?.dtype(),
        Region::Layout => postscript(file)?.layout(),
        Region::Statistics => postscript(file)?.statistics(),
        Region::Footer => postscript(file)?.footer(),
    }?;
    span(file, segment.offset(), segment.length())
}

fn apply_bytes(file: &mut [u8], region: Region, offset: u32, byte: u8) {
    let Some((start, length)) = region_span(file, region) else {
        return;
    };
    if length == 0 {
        return;
    }
    let position = start + usize::try_from(offset).unwrap_or(0) % length;
    if let Some(slot) = file.get_mut(position) {
        *slot ^= byte;
    }
}
