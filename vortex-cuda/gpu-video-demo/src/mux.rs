// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! MPEG-TS muxer for wrapping H.264 NAL units into transport stream packets.
//!
//! Produces 188-byte MPEG-TS packets suitable for SRT transport.

#![allow(clippy::cast_possible_truncation)]

const TS_PACKET_SIZE: usize = 188;
const TS_SYNC_BYTE: u8 = 0x47;

const PAT_PID: u16 = 0x0000;
const PMT_PID: u16 = 0x0100;
const VIDEO_PID: u16 = 0x0101;

const H264_STREAM_TYPE: u8 = 0x1B;

/// MPEG-TS muxer for a single H.264 video stream.
pub struct TsMuxer {
    pat_cc: u8,
    pmt_cc: u8,
    video_cc: u8,
    fps: u32,
}

impl TsMuxer {
    /// Creates a new MPEG-TS muxer.
    pub fn new(fps: u32) -> Self {
        Self {
            pat_cc: 0,
            pmt_cc: 0,
            video_cc: 0,
            fps,
        }
    }

    /// Wraps an H.264 access unit (one frame's NAL units) into MPEG-TS packets.
    ///
    /// Returns a vector of 188-byte TS packets including PAT, PMT, and PES.
    pub fn write_access_unit(&mut self, h264_nals: &[u8], frame_idx: u64) -> Vec<u8> {
        let pts = frame_idx * 90000 / u64::from(self.fps);
        let pcr = pts;

        let mut output = Vec::new();

        // Emit PAT + PMT every frame for easy stream joining
        output.extend_from_slice(&self.build_pat());
        output.extend_from_slice(&self.build_pmt());

        // Build PES packet
        let pes = self.build_pes(h264_nals, pts);

        // Split PES into TS packets
        let mut offset = 0;
        let mut first = true;
        while offset < pes.len() {
            let mut packet = [0xFFu8; TS_PACKET_SIZE];

            // Sync byte
            packet[0] = TS_SYNC_BYTE;

            // PID and flags
            let pusi = if first { 1u8 } else { 0u8 };
            packet[1] = (pusi << 6) | ((VIDEO_PID >> 8) as u8 & 0x1F);
            packet[2] = (VIDEO_PID & 0xFF) as u8;

            let adaptation_field = first; // PCR on first packet
            let cc = self.video_cc;
            self.video_cc = (self.video_cc + 1) & 0x0F;

            let mut header_len = 4;

            if adaptation_field {
                // Adaptation field with PCR
                let af_len = 8; // 1 (flags) + 6 (PCR) + 1 (stuffing flag byte)
                packet[3] = 0x30 | cc; // adaptation_field + payload + cc
                packet[4] = af_len as u8; // adaptation field length
                packet[5] = 0x10; // PCR flag set

                // PCR: 33 bits base + 6 reserved + 9 extension
                let pcr_base = pcr;
                let pcr_ext: u16 = 0;
                packet[6] = ((pcr_base >> 25) & 0xFF) as u8;
                packet[7] = ((pcr_base >> 17) & 0xFF) as u8;
                packet[8] = ((pcr_base >> 9) & 0xFF) as u8;
                packet[9] = ((pcr_base >> 1) & 0xFF) as u8;
                packet[10] = (((pcr_base & 1) << 7) | 0x7E | (u64::from(pcr_ext >> 8) & 1)) as u8;
                packet[11] = (pcr_ext & 0xFF) as u8;

                header_len = 4 + 1 + af_len; // TS header + AF length byte + AF
            } else {
                packet[3] = 0x10 | cc; // payload only + cc
            }

            let payload_space = TS_PACKET_SIZE - header_len;
            let remaining = pes.len() - offset;
            let copy_len = remaining.min(payload_space);

            // If this is the last packet and there's unused space, add stuffing
            if copy_len < payload_space {
                let stuff_len = payload_space - copy_len;
                if !adaptation_field {
                    // Need to add adaptation field for stuffing
                    packet[3] = (packet[3] & 0x0F) | 0x30; // add AF flag
                    if stuff_len == 1 {
                        packet[4] = 0; // AF length = 0
                        let start = 5;
                        packet[start..start + copy_len]
                            .copy_from_slice(&pes[offset..offset + copy_len]);
                    } else {
                        let af_content_len = stuff_len - 1;
                        packet[4] = af_content_len as u8;
                        packet[5] = 0x00; // no flags
                        // Fill remaining AF with 0xFF
                        for b in &mut packet[6..6 + af_content_len.saturating_sub(1)] {
                            *b = 0xFF;
                        }
                        let start = 4 + 1 + af_content_len;
                        packet[start..start + copy_len]
                            .copy_from_slice(&pes[offset..offset + copy_len]);
                    }
                } else {
                    // Already have AF, extend stuffing
                    let existing_af_len = packet[4] as usize;
                    let new_af_len = existing_af_len + stuff_len;
                    packet[4] = new_af_len as u8;
                    // Fill new stuffing bytes
                    let stuff_start = 5 + existing_af_len;
                    for b in &mut packet[stuff_start..stuff_start + stuff_len] {
                        *b = 0xFF;
                    }
                    let start = 5 + new_af_len;
                    packet[start..start + copy_len]
                        .copy_from_slice(&pes[offset..offset + copy_len]);
                }
            } else {
                packet[header_len..header_len + copy_len]
                    .copy_from_slice(&pes[offset..offset + copy_len]);
            }

            output.extend_from_slice(&packet);
            offset += copy_len;
            first = false;
        }

        output
    }

    fn build_pat(&mut self) -> [u8; TS_PACKET_SIZE] {
        let mut packet = [0xFFu8; TS_PACKET_SIZE];
        packet[0] = TS_SYNC_BYTE;
        packet[1] = 0x40; // PUSI + PID 0
        packet[2] = 0x00;
        let cc = self.pat_cc;
        self.pat_cc = (self.pat_cc + 1) & 0x0F;
        packet[3] = 0x10 | cc; // payload only

        // Pointer field
        packet[4] = 0x00;

        // PAT table
        let pat = &mut packet[5..];
        pat[0] = 0x00; // table_id = PAT
        // section_syntax_indicator=1, 0, reserved=11
        let section_len = 9u16; // 5 (header after length) + 4 (CRC)
        pat[1] = 0xB0 | ((section_len >> 8) as u8 & 0x0F);
        pat[2] = (section_len & 0xFF) as u8;
        pat[3] = 0x00; // transport_stream_id
        pat[4] = 0x01;
        pat[5] = 0xC1; // reserved + version=0 + current_next=1
        pat[6] = 0x00; // section_number
        pat[7] = 0x00; // last_section_number
        // Program 1 → PMT PID
        pat[8] = 0x00; // program_number high
        pat[9] = 0x01; // program_number low
        pat[10] = 0xE0 | ((PMT_PID >> 8) as u8 & 0x1F);
        pat[11] = (PMT_PID & 0xFF) as u8;

        // CRC32
        let crc = crc32_mpeg2(&packet[5..5 + 3 + section_len as usize - 4]);
        let crc_offset = 5 + 3 + section_len as usize - 4;
        packet[crc_offset] = ((crc >> 24) & 0xFF) as u8;
        packet[crc_offset + 1] = ((crc >> 16) & 0xFF) as u8;
        packet[crc_offset + 2] = ((crc >> 8) & 0xFF) as u8;
        packet[crc_offset + 3] = (crc & 0xFF) as u8;

        packet
    }

    fn build_pmt(&mut self) -> [u8; TS_PACKET_SIZE] {
        let mut packet = [0xFFu8; TS_PACKET_SIZE];
        packet[0] = TS_SYNC_BYTE;
        packet[1] = 0x40 | ((PMT_PID >> 8) as u8 & 0x1F);
        packet[2] = (PMT_PID & 0xFF) as u8;
        let cc = self.pmt_cc;
        self.pmt_cc = (self.pmt_cc + 1) & 0x0F;
        packet[3] = 0x10 | cc;

        // Pointer field
        packet[4] = 0x00;

        // PMT table
        let pmt = &mut packet[5..];
        pmt[0] = 0x02; // table_id = PMT
        let section_len = 13u16; // header + stream info + CRC
        pmt[1] = 0xB0 | ((section_len >> 8) as u8 & 0x0F);
        pmt[2] = (section_len & 0xFF) as u8;
        pmt[3] = 0x00; // program_number
        pmt[4] = 0x01;
        pmt[5] = 0xC1; // reserved + version=0 + current_next=1
        pmt[6] = 0x00; // section_number
        pmt[7] = 0x00; // last_section_number
        // PCR PID
        pmt[8] = 0xE0 | ((VIDEO_PID >> 8) as u8 & 0x1F);
        pmt[9] = (VIDEO_PID & 0xFF) as u8;
        pmt[10] = 0xF0; // program_info_length = 0
        pmt[11] = 0x00;
        // Stream entry: H.264 video
        pmt[12] = H264_STREAM_TYPE;
        pmt[13] = 0xE0 | ((VIDEO_PID >> 8) as u8 & 0x1F);
        pmt[14] = (VIDEO_PID & 0xFF) as u8;
        pmt[15] = 0xF0; // ES_info_length = 0
        pmt[16] = 0x00;

        // CRC32
        let crc = crc32_mpeg2(&packet[5..5 + 3 + section_len as usize - 4]);
        let crc_offset = 5 + 3 + section_len as usize - 4;
        packet[crc_offset] = ((crc >> 24) & 0xFF) as u8;
        packet[crc_offset + 1] = ((crc >> 16) & 0xFF) as u8;
        packet[crc_offset + 2] = ((crc >> 8) & 0xFF) as u8;
        packet[crc_offset + 3] = (crc & 0xFF) as u8;

        packet
    }

    fn build_pes(&self, h264_nals: &[u8], pts: u64) -> Vec<u8> {
        // PES header: start code (3) + stream_id (1) + length (2) + flags (3) + PTS (5)
        let pes_header_data_len = 5u8; // PTS only
        let pes_header_len = 9usize + pes_header_data_len as usize;
        let pes_packet_len = 3 + pes_header_data_len as usize + h264_nals.len();

        let mut pes = Vec::with_capacity(pes_header_len + h264_nals.len());

        // Start code prefix
        pes.push(0x00);
        pes.push(0x00);
        pes.push(0x01);

        // Stream ID: video stream 0
        pes.push(0xE0);

        // PES packet length (0 = unbounded for video, but we set it if it fits)
        if pes_packet_len <= 0xFFFF {
            #[allow(clippy::cast_possible_truncation)]
            {
                pes.push(((pes_packet_len >> 8) & 0xFF) as u8);
                pes.push((pes_packet_len & 0xFF) as u8);
            }
        } else {
            pes.push(0x00);
            pes.push(0x00);
        }

        // Flags: data_alignment=1, PTS_DTS_flags=10 (PTS only)
        pes.push(0x80); // 10 00 0 0 0 0
        pes.push(0x80); // PTS only
        pes.push(pes_header_data_len);

        // PTS (5 bytes, 33 bits)
        pes.push((0x20 | (((pts >> 30) & 0x07) << 1) | 1) as u8);
        pes.push(((pts >> 22) & 0xFF) as u8);
        pes.push((((pts >> 15) & 0x7F) << 1 | 1) as u8);
        pes.push(((pts >> 7) & 0xFF) as u8);
        pes.push((((pts & 0x7F) << 1) | 1) as u8);

        // H.264 data
        pes.extend_from_slice(h264_nals);

        pes
    }
}

/// MPEG-2 CRC32 calculation.
fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ 0x04C1_1DB7;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}
