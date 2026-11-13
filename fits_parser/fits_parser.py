#!/usr/bin/env python3
"""
Read BTO binary telemetry and write the BTO subset of the COSI archive.

"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

import numpy as np
from astropy.io import fits
from astropy.time import Time


# 1. Constants. Keep firmware-dependent assumptions together for review.
VERSION = "6.0.0"
ARCHIVE_FORMAT = "BTO_BINARY_FITS_V6"
PREVIOUS_ARCHIVE_FORMATS = ("BTO_BINARY_FITS_V3", "BTO_BINARY_FITS_V4", "BTO_BINARY_FITS_V5")
GPS_AT_MET_ZERO = 1_419_724_818       # 2025-01-01 00:00:00 UTC
MJDREFI = 60676
MJDREFF = 69.184 / 86400.0          # The same reference instant, expressed in TT
PPS_HZ = 40_000_000
DWT_HZ = 300_000_000
DWT_CYCLE = 2**32                  # Only the upper 24 DWT bits are in events
DWT_QUANTUM = 256                  # One event-clock step, in 300 MHz cycles
DEADTIME_TICK = 0.5e-6             # seconds per telemetry deadtime tick
MAX_ANCHOR_SPAN = 14.0            # Two absolute anchors resolve almost one full DWT cycle
CLOCK_TOLERANCE = 100e-6           # Allowed fractional clock-rate difference
SYNC = b"\xeb\x90"
SCIENCE_APIDS = (0xD6, 0xD7, 0xD8)
DATA_LENGTHS = {0xD6: (128,), 0xD7: (246,), 0xD8: (82, 84)}

# TIME_QUAL bits. Zero means that the stated timing checks passed.
BAD_CLOCK = 1
SEQUENCE_GAP = 2
NO_ANCHOR = 4
BAD_MARKER = 8
BAD_DWT = 16
UNCERTAIN_RECORD = 32
BAD_DEADTIME = 64
COUNT_MISMATCH = 128              # Timed, but heartbeat event count disagrees
BRIDGED_ANCHOR = 256              # Timed across an inconsistent intermediate clock sample

# Python < 3.10 compatibility: dataclass(slots=True) was added in Python 3.10.
DATACLASS_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


class ParseError(ValueError):
    """Input cannot be decoded without guessing or losing bytes."""


@dataclass(**DATACLASS_SLOTS)
class Packet:
    source: int
    offset: int                    # First byte in the original binary file
    header_offset: int             # Absolute location of the CCSDS header
    apid: int
    sequence: int
    data: bytes                    # Starts AFTER the six-byte primary header
    gap: bool = False

    @property
    def detector(self):
        return self.data[8]

    @property
    def time(self):
        seconds, ticks = struct.unpack_from(">II", self.data)
        if GPS_AT_MET_ZERO <= seconds < 0xFFFFFFFF and 0 <= ticks < PPS_HZ:
            return (seconds - GPS_AT_MET_ZERO) + ticks / PPS_HZ
        return float("nan")


# 2. Binary framing. Never search inside payloads for a new sync word.
def xor_checksum(body):
    """BTO XOR16, seeded with FFFF; excludes EB90 and the checksum itself."""
    if len(body) % 2:
        raise ParseError("XOR16 requires an even number of packet bytes")
    value = 0xFFFF
    for (word,) in struct.iter_unpack(">H", body):
        value ^= word
    return value


def read_packets(path, source=0, framing="auto", checksum="absent"):
    """Yield length-delimited CCSDS packets from a .bin file.

    Total CCSDS bytes = 6 + (packet_length + 1). The uploaded binaries have
    neither EB90 nor an external checksum. Both are supported explicitly for
    other captures. A truncated packet is an error, including at end of file.
    """
    if framing not in ("auto", "ccsds", "sync") or checksum not in ("absent", "required"):
        raise ValueError("Invalid framing or checksum option")
    with Path(path).open("rb") as stream:
        first = stream.read(2)
        stream.seek(0)
        has_sync = first == SYNC if framing == "auto" else framing == "sync"
        while True:
            offset = stream.tell()
            prefix = stream.read(2 if has_sync else 6)
            if not prefix:
                return
            if has_sync and prefix != SYNC:
                raise ParseError(f"{path}, byte {offset}: expected EB90")
            header = stream.read(6) if has_sync else prefix
            if len(header) != 6:
                raise ParseError(f"{path}, byte {offset}: truncated primary header")
            packet_id, sequence, length = struct.unpack(">HHH", header)
            apid = packet_id & 0x7FF
            if packet_id >> 13:
                raise ParseError(f"{path}, byte {offset}: unsupported CCSDS version")
            if apid in SCIENCE_APIDS:
                if packet_id & 0x1800 != 0x0800 or sequence >> 14 != 3:
                    raise ParseError(f"{path}, byte {offset}: invalid BTO telemetry header")
                if length + 1 not in DATA_LENGTHS[apid]:
                    raise ParseError(f"{path}, byte {offset}: APID {apid:#x} data length {length + 1}")
            data = stream.read(length + 1)
            if len(data) != length + 1:
                raise ParseError(f"{path}, byte {offset}: truncated APID {apid:#x} payload")
            if checksum == "required":
                check = stream.read(2)
                if len(check) != 2 or int.from_bytes(check, "big") != xor_checksum(header + data):
                    raise ParseError(f"{path}, byte {offset}: missing or incorrect XOR16 checksum")
            if apid in SCIENCE_APIDS and data[8] not in (0, 1):
                raise ParseError(f"{path}, byte {offset}: invalid detector ID {data[8]}")
            yield Packet(source, offset, offset + (2 if has_sync else 0),
                         apid, sequence & 0x3FFF, data)


# 3. Common time and array helpers.
def mission_time(seconds, ticks=0):
    """GPS/PPS -> elapsed SI seconds from the ICD's mission reference instant."""
    sec, sub = np.broadcast_arrays(np.asarray(seconds, dtype=np.int64),
                                   np.asarray(ticks, dtype=np.int64))
    valid = (sec >= GPS_AT_MET_ZERO) & (sec < 0xFFFFFFFF) & (sub >= 0) & (sub < PPS_HZ)
    return np.where(valid, (sec - GPS_AT_MET_ZERO) + sub / PPS_HZ, np.nan)


def utc_string(met):
    return Time(GPS_AT_MET_ZERO + float(met), format="gps").utc.isot


def date_key(met):
    return utc_string(met)[:10].replace("-", "") if np.isfinite(met) else "undated"


def payload_array(packets, width=None):
    """A byte matrix is convenient for fixed-layout telemetry packets."""
    if width is None:
        raw = b"".join(p.data for p in packets)
    else:
        raw = b"".join(p.data[:width] for p in packets)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(packets), -1)


def unsigned_field(data, offset, width, count=1):
    """Read big-endian unsigned fields without any signed integer conversion."""
    field = np.ascontiguousarray(data[:, offset:offset + width * count])
    values = field.view(f">u{width}").reshape(len(data), count).astype(np.int64)
    return values[:, 0] if count == 1 else values


def provenance(packets):
    header_bytes = payload_array(packets, width=8)
    pkt_sec = unsigned_field(header_bytes, 0, 4)
    pkt_sub = unsigned_field(header_bytes, 4, 4)
    return {
        "PKT_CNT": np.array([p.sequence for p in packets], dtype=np.int16),
        "SOURCE_ID": np.array([p.source for p in packets], dtype=np.int32),
        "BYTE_OFFSET": np.array([p.header_offset for p in packets], dtype=np.int64),
        "PKT_TIME": mission_time(pkt_sec, pkt_sub),
    }


def time_order(values):
    """Stable chronological order; unknown times remain at the end."""
    return np.argsort(np.where(np.isfinite(values), values, np.inf), kind="stable")


def subset(table, indices):
    return {key: value[indices] for key, value in table.items()}


# 4. D6 histograms and 100 ms counters.
def decode_histograms(packets):
    d = payload_array(packets)
    seconds, ticks = unsigned_field(d, 71, 4), unsigned_field(d, 75, 4)
    dead_ticks = unsigned_field(d, 67, 4)
    deadtime = dead_ticks * DEADTIME_TICK
    time = mission_time(seconds, ticks)
    quality = np.where(np.isfinite(time), 0, BAD_CLOCK)
    quality |= np.array([SEQUENCE_GAP if p.gap else 0 for p in packets])
    quality |= np.where(deadtime <= 1.0, 0, BAD_DEADTIME)
    hist = dict(TIME=time, **provenance(packets),
                TIMEDEL=np.where(deadtime <= 1.0, 1.0 - deadtime, np.nan),
                COUNT=unsigned_field(d, 9, 2, 29), TIME_QUAL=quality,
                DEADTIME=deadtime, DEAD_TICKS=dead_ticks,
                HIST_SEC=seconds, HIST_TICKS=ticks, BINMODE=d[:, 127].astype(int))

    # Unlike the explicitly swapped histogram fields, these eight bytes were
    # copied from an MCU structure: little-endian PPS, then 24-bit ticks + FF.
    lc_seconds = np.ascontiguousarray(d[:, 119:123]).view("<u4").ravel().astype(np.int64)
    lc_ticks = (d[:, 123].astype(np.int64) + (d[:, 124].astype(np.int64) << 8)
                + (d[:, 125].astype(np.int64) << 16)) << 8
    lc_end = mission_time(lc_seconds, lc_ticks)
    lc_end[d[:, 126] != 255] = np.nan
    blocks = d[:, 79:119].reshape(-1, 10, 4).astype(np.int64)
    counters = np.stack((blocks[:, :, 0] * 256 + blocks[:, :, 1],
                         blocks[:, :, 2], blocks[:, :, 3]), axis=2).reshape(-1, 3)

    # The samples contain cumulative counters. The first bin needs a previous
    # packet; later bins use the preceding sample in their own packet.
    counts = (counters - np.roll(counters, 1, axis=0)) % [65536, 256, 256]
    count_quality = np.zeros(len(counts), dtype=np.int16)
    bad_first = np.ones(len(packets), dtype=bool)
    if len(packets) > 1:
        bad_first[1:] = (~np.isfinite(lc_end[1:] - lc_end[:-1])
                         | (np.abs(lc_end[1:] - lc_end[:-1] - 1.0) > 0.01)
                         | np.array([p.gap for p in packets[1:]]))
    unknown = np.flatnonzero(bad_first) * 10
    counts[unknown] = -1
    count_quality[unknown] = 1
    lc_time = (lc_end[:, None] - np.arange(9, -1, -1)[None, :] * 0.1).ravel()
    lc = {key: np.repeat(value, 10) for key, value in provenance(packets).items()}
    lc.update(TIME=lc_time, TIMEDEL=np.full(len(counts), 0.1), COUNT=counts,
              COUNTER_RAW=counters, COUNT_QUAL=count_quality,
              TIME_QUAL=np.where(np.isfinite(lc_time), 0, BAD_CLOCK),
              SAMPLE=np.tile(np.arange(10, dtype=np.int16), len(packets)))
    return hist, lc


# 5. D8 housekeeping. Offsets are relative to the CCSDS DATA field.
HK_FIELDS = [
    ("BTO_MODE", 9, "B"), ("UPTIME_TICKS", 10, "I"),
    ("CMD_CNT", 14, "H"), ("FAIL_CMD_CNT", 16, "H"), ("CS_ERR_CNT", 18, "H"),
    ("FSW_VER", 20, "B"), ("WD_RESET", 21, "B"), ("RESET_REASON", 22, "B"),
    ("FAULT_STATUS", 23, "B"), ("DET_PWR_STATUS", 24, "B"), ("ANALOG_STATUS", 25, "B"),
    ("HIST_WPTR", 26, "H"), ("HIST_RPTR", 28, "H"),
    ("HIST_NAND_WPTR", 30, "I"), ("HIST_NAND_RPTR", 34, "I"),
    ("PHOT_NAND_WPTR", 38, "I"), ("PHOT_NAND_RPTR", 42, "I"),
    ("FLASH_ERASE_FAIL", 46, "H"), ("LAST_GRB_TIME_RAW", 48, "I"),
    ("LAST_GRB_NAND", 52, "I"), ("LAST_GRB_ID", 56, "I"),
    ("ZERO_CROSSING_COUNT", 60, "H"), ("SUPPER_UPPER_COUNT", 62, "H"),
    ("TEMP_EXT_ADC", 64, "H"), ("TEMP_DET_1_ADC", 66, "H"), ("TEMP_DET_2_ADC", 68, "H"),
    ("AB1_VMON_10V_P_ADC", 70, "H"), ("AB1_VMON_10V_N_ADC", 72, "H"),
    ("AB2_VMON_10V_P_ADC", 74, "H"), ("AB2_VMON_10V_N_ADC", 76, "H"),
    ("DB1_IMON_5V_ADC", 78, "H"), ("DB2_IMON_5V_ADC", 80, "H"),
]
HK_WIDTHS = {"B": 1, "H": 2, "I": 4}
SENSORS = [(name[:-4], "mA" if name.startswith("DB") else
            "C" if name.startswith("TEMP") else "V")
           for name, _, _ in HK_FIELDS if name.endswith("_ADC")]


def decode_housekeeping(packets, calibration):
    d = payload_array(packets, width=82)
    result = provenance(packets)
    result["TIME"] = result["PKT_TIME"].copy()
    result["BTO_ID"] = d[:, 8].copy()
    for name, offset, code in HK_FIELDS:
        result[name] = unsigned_field(d, offset, HK_WIDTHS[code])
    result["SPARE_RAW"] = np.array([int.from_bytes(p.data[82:84], "big")
                                     if len(p.data) == 84 else -1 for p in packets], dtype=np.int64)
    result["LAST_GRB_TIME"] = mission_time(result["LAST_GRB_TIME_RAW"])
    result["TIME_QUAL"] = np.where(np.isfinite(result["TIME"]), 0, BAD_CLOCK)
    result["TIME_QUAL"] |= np.array([SEQUENCE_GAP if p.gap else 0 for p in packets])
    for name, _ in SENSORS:
        result[name] = np.full(len(packets), np.nan, dtype=np.float32)
        for detector in (0, 1):
            coefficients = detector_calibration(calibration, detector).get("housekeeping", {}).get(name)
            if coefficients is not None:
                chosen = result["BTO_ID"] == detector
                slope, intercept = coefficients
                result[name][chosen] = result[name + "_ADC"][chosen] * slope + intercept
    return result


# 6. D7 photon records. Each detector is decoded independently.
def low_word_first(words, offset):
    """Marker 32-bit values store the LOW 16-bit word before the HIGH word."""
    return int(words[offset]) | (int(words[offset + 1]) << 16)


def decode_events(packets):
    """Decode the current three-record heartbeat format used by the samples.

    START contains trigger flags and an ID, NOT a timestamp. Events receive
    TIME only between consistent heartbeat or END anchors. Counter wrap cannot
    be resolved by simply adding the nearest modulo difference repeatedly.
    """
    for p in packets:
        if p.data[13] != 255:
            raise ParseError(f"D7 source {p.source}, byte {p.offset}: missing record filler FF")
    words = np.frombuffer(b"".join(p.data[14:] for p in packets), dtype=">u2").reshape(-1, 4)
    n = len(words)
    packet_index = np.repeat(np.arange(len(packets)), 29)
    record_index = np.tile(np.arange(29), len(packets))
    event_mask = ~np.all(words == 0, axis=1)
    marker_mask = np.zeros(n, dtype=bool)
    erased = np.all(words == 65535, axis=1)
    event_mask[erased] = False
    starts = (words[:, 0] == 0x0123) & (words[:, 1] == 0x4567)
    heartbeats = (words[:, 0] == 65535) & (words[:, 1] == 65535) & ~erased
    ends = (words[:, 0] == 0xABCD) & (words[:, 1] == 0xABCD)
    continuations = (words[:, 0] == 0xABCD) & np.isin(words[:, 1], [0xEF00, 0xEFFF])
    gaps = np.zeros(n, dtype=bool)
    gaps[np.arange(len(packets)) * 29] = [p.gap for p in packets]
    segment_break = gaps | erased | starts
    group_break = starts.copy()
    group_break[0] = True
    quality = np.full(n, NO_ANCHOR, dtype=np.int16)
    marker_rows = []
    anchors = []

    for pos in np.flatnonzero(starts | heartbeats | ends | continuations | erased):
        if not event_mask[pos] and not erased[pos]:
            continue                     # Already consumed as a continuation
        packet = packets[packet_index[pos]]
        marker = dict(source=packet.source, offset=int(packet.header_offset + 20 + record_index[pos] * 8),
                      kind="UNKNOWN", records=1, trigger_id=-1)
        take = 1
        if starts[pos]:
            marker.update(kind="START", trigger_id=int(words[pos, 3]), flags=int(words[pos, 2]))
        elif erased[pos]:
            marker["kind"] = "ERASED"
        elif heartbeats[pos] or ends[pos]:
            complete = pos + 2 < n and not np.any(gaps[pos + 1:pos + 3])
            if complete and heartbeats[pos]:
                sec = low_word_first(words[pos + 1], 0)
                ticks = low_word_first(words[pos + 1], 2)
                complete = not np.any(starts[pos + 1:pos + 3] | ends[pos + 1:pos + 3]
                                      | heartbeats[pos + 1:pos + 3] | erased[pos + 1:pos + 3])
            elif complete:
                sec = low_word_first(words[pos], 2)
                ticks = low_word_first(words[pos + 1], 2)
                complete = bool(continuations[pos + 1] and not
                                (starts[pos + 2] or heartbeats[pos + 2] or ends[pos + 2] or erased[pos + 2]))
            if complete:
                value = float(mission_time(sec, ticks))
                complete = np.isfinite(value) and (not np.isfinite(packet.time) or value <= packet.time + 1.0)
            if complete:
                take = 3
                marker.update(kind="HEARTBEAT" if heartbeats[pos] else "END",
                              gps_seconds=sec, pps_ticks=ticks,
                              uptime_ticks=low_word_first(words[pos + 2], 0),
                              dwt=low_word_first(words[pos + 2], 2), time=value)
                if heartbeats[pos]:
                    marker["event_count"] = low_word_first(words[pos], 2)
                anchors.append(dict(marker, position=int(pos)))
            else:
                marker["kind"] = "INCOMPLETE_OR_UNSUPPORTED_MARKER"
                segment_break[pos] = True
                quality[pos + 1:min(n, pos + 3)] |= UNCERTAIN_RECORD | BAD_MARKER
                segment_break[min(n - 1, pos + 2)] = True
        else:
            marker["kind"] = "ORPHAN_END_CONTINUATION"
            quality[pos + 1:min(n, pos + 2)] |= UNCERTAIN_RECORD | BAD_MARKER
            segment_break[pos] = True
        marker["records"] = take
        marker["raw_hex"] = words[pos:pos + take].tobytes().hex()
        marker_rows.append(marker)
        event_mask[pos:pos + take] = False
        marker_mask[pos:pos + take] = True
        if ends[pos] or continuations[pos]:
            if pos + take < n:
                group_break[pos + take] = True
                segment_break[pos + take] = True

    # A break in telemetry also prevents orphan marker continuations from
    # acquiring a valid TIME by an accidental subsequent clock match.
    for pos in np.flatnonzero(gaps):
        quality[pos:min(n, pos + 2)] |= SEQUENCE_GAP | UNCERTAIN_RECORD
    segment = np.cumsum(segment_break)
    group = np.cumsum(group_break) - 1
    group_trigger = np.full(int(group[-1]) + 1, -1, dtype=np.int64)
    group_trigger[group[starts]] = words[starts, 3]
    dwt = (words[:, 1].astype(np.int64) << 16) | (words[:, 2].astype(np.int64) & 0xFF00)
    times = np.full(n, np.nan)
    timing_pair = np.full(n, -1, dtype=np.int32)
    pairs = []
    rejected = Counter()

    # A pair of absolute clock anchors bounds the event time to ONE DWT cycle.
    # The old half-cycle limit was needed for nearest-wrap guesses, not for
    # this bracketed reconstruction. Try a wider bracket when a single marker
    # has an inconsistent clock sample; never bridge a telemetry/reset break.
    # Event counts are a diagnostic, not a clock: missing one photon cannot
    # change the timestamp encoded in every other photon's DWT field.
    candidates = []
    for left_index, left in enumerate(anchors[:-1]):
        for right_index in range(left_index + 1, len(anchors)):
            right = anchors[right_index]
            if segment[left["position"]] != segment[right["position"]]:
                break
            elapsed = ((right["gps_seconds"] - left["gps_seconds"])
                       + (right["pps_ticks"] - left["pps_ticks"]) / PPS_HZ)
            if elapsed > MAX_ANCHOR_SPAN:
                break
            if elapsed <= 0:
                continue
            candidates.append((right_index - left_index, left_index, right_index, elapsed))

    # Closest brackets win. Wider pairs only fill rows still missing a time.
    # This also handles an interrupt writing a photon just across a marker.
    skipped_anchors = set()
    count_warnings = []
    for separation, left_index, right_index, elapsed in sorted(candidates):
        left, right = anchors[left_index], anchors[right_index]
        lo, hi = left["position"] + 3, right["position"]
        ids = np.flatnonzero(event_mask[lo:hi]) + lo
        if not np.any(~np.isfinite(times[ids])):
            continue
        span = (right["dwt"] - left["dwt"]) % DWT_CYCLE
        uptime = (right["uptime_ticks"] - left["uptime_ticks"]) % 2**32 / 32.0
        if abs(uptime - elapsed) > 0.07 + elapsed * CLOCK_TOLERANCE:
            rejected["uptime_mismatch"] += 1
            continue
        # Allow two microseconds for quantization/readout on very short pairs.
        if span <= 0 or abs(span / DWT_HZ - elapsed) > 2e-6 + elapsed * CLOCK_TOLERANCE:
            rejected["clock_mismatch"] += 1
            continue
        delta = (dwt[ids] - left["dwt"]) % DWT_CYCLE
        delta[delta >= DWT_CYCLE - DWT_QUANTUM] = 0
        if (np.any(delta > span + DWT_QUANTUM) or np.any(np.diff(delta) < 0)
                or np.any(quality[ids] & UNCERTAIN_RECORD)):
            rejected["event_clock_or_record_mismatch"] += 1
            continue
        flags = 0
        if "event_count" in left and "event_count" in right:
            expected = (right["event_count"] - left["event_count"]) % 2**32
            if expected != len(ids):
                flags |= COUNT_MISMATCH
                count_warnings.append(dict(left=left["offset"], right=right["offset"],
                                           expected=expected, received=len(ids)))
        if right_index > left_index + 1:
            flags |= BRIDGED_ANCHOR
            skipped_anchors.update(range(left_index + 1, right_index))
        untimed = ~np.isfinite(times[ids])
        chosen = ids[untimed]
        times[chosen] = left["time"] + np.minimum(delta[untimed], span) * (elapsed / span)
        quality[chosen] = flags
        timing_pair[chosen] = len(pairs)
        if len(chosen):
            pairs.append(dict(start=left["time"], stop=right["time"], events=len(chosen),
                              left_offset=left["offset"], right_offset=right["offset"],
                              source=left["source"], right_source=right["source"], flags=flags,
                              clock_ppm=(span / elapsed / DWT_HZ - 1) * 1e6))

    selected = np.flatnonzero(event_mask)
    pids = packet_index[selected]
    base = provenance(packets)
    result = {key: value[pids] for key, value in base.items()}
    result["BYTE_OFFSET"] += 20 + record_index[selected] * 8
    result.update(TIME=times[selected], PHA=(words[selected, 0] & 0xFFF).astype(np.int16),
                  ENERGY=np.full(len(selected), np.nan, dtype=np.float32),
                  DEADTIME=words[selected, 3].astype(float) * DEADTIME_TICK,
                  TIME_QUAL=quality[selected], TIMING_PAIR=timing_pair[selected],
                  RAW_ADC=words[selected, 0].astype(np.int64),
                  DWT=dwt[selected], TSU_TICKS=(words[selected, 2] & 255).astype(np.int16),
                  DEAD_TICKS=words[selected, 3].astype(np.int64),
                  FLAGS=(words[selected, 0] >> 12).astype(np.uint8), GROUP=group[selected])
    report = dict(input_records=n, events=len(selected),
                  padding_records=int(np.sum(np.all(words == 0, axis=1) & ~marker_mask)),
                  marker_records=sum(m["records"] for m in marker_rows),
                  timed_events=int(np.sum(np.isfinite(result["TIME"]))),
                  markers=marker_rows, accepted_pairs=pairs, rejected_pairs=dict(rejected),
                  bridged_anchor_offsets=[anchors[i]["offset"] for i in sorted(skipped_anchors)],
                  count_warnings=count_warnings)
    if report["events"] + report["padding_records"] + report["marker_records"] != n:
        raise ParseError("Photon record accounting failed")
    return result, group_trigger, report


# 7. Supplied calibration. Every pair is [slope, intercept], so y = slope*x + intercept.
# Use the precise coefficients in the upper spreadsheet, with the C/V/mA
# units in its lower section and the ICD. The upper K/A labels conflict.
# The green "alt" rows directly calibrate the negative rails. The red rows
# describe a voltage delta, not the negative-rail columns required by the ICD.
HOUSEKEEPING_CALIBRATION = {
    "TEMP_DET_1": (0.5432, -268.3),       # Detector Temp #1; degrees Celsius
    "TEMP_DET_2": (0.5437, -268.7),       # Detector Temp #2; degrees Celsius
    "AB1_VMON_10V_P": (0.0045, -2.4681), # volts
    "AB1_VMON_10V_N": (0.0051, -23.963), # volts; green direct -10V row
    "AB2_VMON_10V_P": (0.0048, -3.649),  # volts
    "AB2_VMON_10V_N": (0.0051, -23.906), # volts; green direct -10V row
    "DB1_IMON_5V": (0.1635, -0.7446),    # milliamperes
    "DB2_IMON_5V": (0.1638, -0.9634),    # milliamperes
}
# TEMP_EXT has no supplied calibration; its raw ADC remains available.


def default_calibration():
    """Return independent, editable calibration dictionaries for both BTOs."""
    return {
        "caldb_version": "NONE",  # User-supplied coefficients, not a CALDB release
        "calibration_id": "BTO_17C_20261113",
        "source": "User-supplied housekeeping sheet and separate BTO1/BTO2 17C energy fits",
        "notes": [
            "Packet ID 0 is BTO1; packet ID 1 is BTO2. Energy fits were measured at 17 C.",
            "No temperature-dependent gain correction or validated ADC range was supplied.",
            "Energy coefficient uncertainties are retained; covariance is unknown.",
            "HK units follow the lower sheet and ICD: C, V, mA; upper K/A labels conflict.",
            "Negative rails use green direct -10V rows, not red delta rows.",
            "TEMP_EXT and the 29-channel histogram boundaries were not supplied.",
        ],
        "detectors": {
            str(detector + 1): {
                "energy": [[0.819876, 3.0631], [0.901133, 0.3974]][detector],
                "energy_uncertainty": [[0.000601, 0.5580], [0.000661, 0.4637]][detector],
                "energy_source": f"BTO{detector + 1} at 17 C",
                "energy_temperature_c": 17.0,
                "energy_reduced_chi2": [28.2616, 11.9507][detector],
                "housekeeping": {name: list(pair) for name, pair in HOUSEKEEPING_CALIBRATION.items()},
            }
            for detector in (0, 1)
        },
    }


def detector_calibration(calibration, detector):
    return calibration.get("detectors", {}).get(str(detector + 1), {}) if detector in (0, 1) else {}


def read_calibration(path=None):
    """Use the supplied defaults, or replace them with an explicit JSON file."""
    data = default_calibration() if path is None else json.loads(Path(path).read_text())
    return validate_calibration(data)


def validate_calibration(data):
    """Reject malformed coefficients before creating an output archive."""
    if not isinstance(data, dict):
        raise ParseError("Calibration must be a JSON object")
    if not data:  # An explicit empty object disables all calibration.
        return data
    if not isinstance(data.get("caldb_version"), str) or not data["caldb_version"]:
        raise ParseError("Calibration needs a nonempty caldb_version")
    if not isinstance(data.get("detectors", {}), dict):
        raise ParseError("Calibration detectors must be an object")
    if set(data.get("detectors", {})) - {"1", "2"}:
        raise ParseError("Calibration detector keys must be 1 (raw ID 0) or 2 (raw ID 1)")
    for detector, values in data.get("detectors", {}).items():
        if not isinstance(values, dict):
            raise ParseError(f"Detector {detector}: calibration must be an object")
        if "energy" in values:
            coefficients = np.asarray(values["energy"], dtype=float)
            if coefficients.shape != (2,) or not np.all(np.isfinite(coefficients)) or coefficients[0] <= 0:
                raise ParseError(f"Detector {detector}: energy needs [positive slope, intercept]")
        if "energy_uncertainty" in values:
            errors = np.asarray(values["energy_uncertainty"], dtype=float)
            if ("energy" not in values or errors.shape != (2,)
                    or not np.all(np.isfinite(errors)) or np.any(errors < 0)):
                raise ParseError(f"Detector {detector}: energy_uncertainty needs two nonnegative errors")
        if not isinstance(values.get("housekeeping", {}), dict):
            raise ParseError(f"Detector {detector}: housekeeping must be an object")
        for name, coefficients in values.get("housekeeping", {}).items():
            if name not in dict(SENSORS) or np.shape(coefficients) != (2,) or not np.all(np.isfinite(coefficients)):
                raise ParseError(f"Detector {detector}: invalid housekeeping calibration {name}")
        if not isinstance(values.get("histogram_modes", {}), dict):
            raise ParseError(f"Detector {detector}: histogram_modes must be an object")
        for mode, band in values.get("histogram_modes", {}).items():
            if not isinstance(band, dict):
                raise ParseError(f"Detector {detector}, mode {mode}: bin mapping must be an object")
            edges = np.asarray(band.get("edges", []), dtype=float)
            if (mode not in ("0", "1") or band.get("type") not in ("LINEAR", "LOG")
                    or edges.shape != (30,) or np.any(~np.isfinite(edges))
                    or np.any(edges != np.floor(edges)) or np.any(np.diff(edges) <= 0)
                    or edges[0] < 0 or edges[-1] > 4096):
                raise ParseError(f"Detector {detector}, mode {mode}: need 30 increasing ADC edges and type")
    return data


# 8. FITS columns and metadata. Original raw values are preserved in L0.
def column(name, values, form, unit=None, unsigned=False, null=None):
    options = dict(name=name, array=values, format=form)
    if unit:
        options["unit"] = unit
    if unsigned:
        bits = 16 if form.endswith("I") else 32
        options.update(array=np.asarray(values, dtype=f"uint{bits}"), bzero=2**(bits - 1))
    if null is not None:
        options["null"] = null
    return fits.Column(**options)


def make_table(name, data, schema):
    """Schema entries: column name, FITS format, unit, unsigned flag."""
    return fits.BinTableHDU.from_columns([
        column(key, data[key], form, unit, unsigned,
               -1 if key == "COUNT" and form == "3J" else None)
        for key, form, unit, unsigned in schema], name=name)


# Only the columns in ICD tables 6.2d, 6.2l and 6.2m belong in L1.
# Extra decoded fields remain available from the original L0 bytes.
HIST_SCHEMA = [("TIME", "1D", "s", False), ("PKT_CNT", "1I", None, False),
               ("TIMEDEL", "1D", "s", False), ("COUNT", "29I", "count", True)]
CTIME_SCHEMA = [("TIME", "1D", "s", False), ("PKT_CNT", "1I", None, False),
                ("TIMEDEL", "1D", "s", False), ("COUNT", "3J", "count", False)]
# 3J retains all unsigned 16-bit increments plus -1 (missing predecessor).
# The column count/order is exactly the ICD's; this width exception is explicit.
EVENT_SCHEMA = [("TIME", "1D", "s", False), ("PKT_CNT", "1I", None, False),
                ("PHA", "1I", "chan", False), ("ENERGY", "1E", "keV", False),
                ("DEADTIME", "1D", "s", False)]


def set_headers(hdu, times, detector, obs_id, calibration, interval=0.0):
    h = hdu.header
    h.update(TELESCOP="COSI", INSTRUME="BTO", OBS_ID=obs_id, ORIGIN="SSL",
             CREATOR=f"fits_parser.py {VERSION}", PROCVER=VERSION, SOFTVER=VERSION,
             TLM2FITS=VERSION, CALDBVER=calibration.get("caldb_version", "NONE"),
             SEQNUM=1, SEQPNUM=1, TIMESYS="TT", TIMEUNIT="s", MJDREFI=MJDREFI,
             MJDREFF=MJDREFF, TIMEREF="LOCAL", TASSIGN="SATELLITE", CLOCKAPP=False,
             DATE=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
    h["DATLEVEL"] = ("L1", "Unified L1; ICD science table columns")
    h["CALIBID"] = (calibration.get("calibration_id", "CUSTOM" if calibration else "NONE"),
                    "Calibration identifier; independent of CALDBVER")
    h["DETNAM"] = f"BTO{detector + 1}" if detector in (0, 1) else "BTO1,BTO2"
    detcal = detector_calibration(calibration, detector)
    if "energy" in detcal:
        h["ECSLOPE"] = (float(detcal["energy"][0]), "Energy slope in keV/ADU")
        h["ECOFFSET"] = (float(detcal["energy"][1]), "Energy intercept in keV")
        h["ECALNAME"] = detcal.get("energy_source", "CUSTOM")
        if "energy_temperature_c" in detcal:
            h["ECALTEMP"] = (float(detcal["energy_temperature_c"]), "Energy calibration temperature in C")
        if "energy_reduced_chi2" in detcal:
            h["ECALCHI2"] = (float(detcal["energy_reduced_chi2"]), "Supplied energy-fit reduced chi-square")
        if "energy_uncertainty" in detcal:
            h["ECSLPERR"] = (float(detcal["energy_uncertainty"][0]), "Supplied slope uncertainty in keV/ADU")
            h["ECOFFERR"] = (float(detcal["energy_uncertainty"][1]), "Supplied intercept uncertainty in keV")
    for note in calibration.get("notes", []):
        h["HISTORY"] = note
    h.comments["MJDREFI"] = "Reference instant: 2025-01-01 00:00:00 UTC"
    finite = np.asarray(times)[np.isfinite(times)]
    if len(finite):
        h["TSTART"] = float(finite.min() - interval)
        h["TSTOP"] = float(finite.max())
        h["DATE-OBS"] = utc_string(h["TSTART"])
        h["DATE-END"] = utc_string(h["TSTOP"])
    else:
        h["TIMESTAT"] = "UNRESOLVED"
    if isinstance(hdu, fits.BinTableHDU):
        h["HDUCLASS"] = "OGIP"
        h["HDUCLAS1"] = "TEMPORALDATA"
        if "TIME" in hdu.columns.names:
            h.comments[f"TTYPE{hdu.columns.names.index('TIME') + 1}"] = "SI seconds since 2025-01-01 00:00:00 UTC"
    h["HISTORY"] = "Archive ICD 2026-11-10; user requested BTO1/BTO2 labels and unified L1."
    h["HISTORY"] = "Original L0 binaries and source offsets are mapped in manifest.json."


def observation_gti(times, name="GTI", interval=0.0):
    """Observation start/stop in MET, without claiming science screening.

    Event bounds are the first/last finite event times. For binned data,
    include the full nominal width of the first end-stamped bin. Missing
    scientific times never acquire a GTI from packet transmission time.
    """
    finite = np.asarray(times)[np.isfinite(times)]
    start = [float(finite.min() - interval)] if len(finite) else []
    stop = [float(finite.max())] if len(finite) else []
    hdu = fits.BinTableHDU.from_columns([
        column("START", start, "1D", "s"), column("STOP", stop, "1D", "s")], name=name)
    hdu.header.update(HDUCLASS="OGIP", HDUCLAS1="GTI", HDUCLAS2="STANDARD",
                      GTISTAT="OBSERVATION" if len(finite) else "UNRESOLVED",
                      OBSERVER="John Tomsick", DATAMODE="UNSCREENED")
    hdu.header["COMMENT"] = "GTI gives the observation span in MET, not quality-screened intervals."
    if not len(finite):
        hdu.header["COMMENT"] = "No finite scientific TIME: observation bounds are unavailable."
    return hdu


def energy_bands(detector, modes, calibration):
    detcal = detector_calibration(calibration, detector)
    mode = str(int(modes[0])) if len(modes) == 1 else "mixed"
    band = detcal.get("histogram_modes", {}).get(mode, {})
    edges = np.array(band.get("edges", []), dtype=int)
    lower = edges[:-1] if len(edges) else np.full(29, -1)
    upper = edges[1:] - 1 if len(edges) else np.full(29, -1)
    e_min, e_max = np.full(29, np.nan), np.full(29, np.nan)
    if len(edges) and "energy" in detcal:
        slope, intercept = detcal["energy"]
        e_min = slope * (edges[:-1] - 0.5) + intercept
        e_max = slope * (edges[1:] - 0.5) + intercept
    hdu = fits.BinTableHDU.from_columns([
        column("MINCHAN", lower, "1I", null=-1), column("MAXCHAN", upper, "1I", null=-1),
        column("E_MIN", e_min, "1E", "keV"), column("E_MAX", e_max, "1E", "keV")],
        name=f"CSPEC_ENEBAND{detector + 1}")
    hdu.header.update(HDUCLASS="OGIP", HDUCLAS1="RESPONSE", HDUCLAS2="EBOUNDS",
                      BINTYPE=band.get("type", "UNKNOWN"),
                      CALSTAT="APPLIED" if np.all(np.isfinite(e_min)) else "UNAVAILABLE")
    return hdu


def science_path(root, day, subdirectory, filename):
    """The archive ICD's monthly/day tree, containing only BTO products."""
    if day == "undated":
        raise ParseError("Cannot write an ICD observation without a usable date")
    month = day[:4] + "_" + day[4:6]
    return root / "cosi" / "data" / "obs" / month / day / subdirectory / filename


def source_ids(table):
    return [int(value) for value in np.unique(table["SOURCE_ID"])]


def retain_undated(skipped, kind, table):
    """Keep undatable records in L0 without inventing a directory or date."""
    skipped.append(dict(kind=kind, rows=len(table["TIME"]), source_ids=source_ids(table),
                        reason="No usable observation date; original records retained in L0"))


def write_fits(path, hdus):
    """A failed write is an exception; it is never reported as a success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with fits.HDUList(hdus) as product:
        product.writeto(path, checksum=True, output_verify="exception")


def date_groups(times, fallback):
    routed = np.where(np.isfinite(times), times, fallback)
    # Vectorized UTC conversion is much faster than one Astropy call per row.
    days = np.full(len(routed), "undated", dtype="U8")
    good = np.isfinite(routed)
    if np.any(good):
        dates = Time(GPS_AT_MET_ZERO + routed[good], format="gps").utc.isot
        days[good] = [d[:10].replace("-", "") for d in dates]
    for day in np.unique(days):
        yield str(day), np.flatnonzero(days == day)


def write_histograms(root, packets, detector, calibration, products, skipped):
    hist, lc = decode_histograms(packets)
    for day, ids in date_groups(hist["TIME"], hist["PKT_TIME"]):
        data = subset(hist, ids)
        if day == "undated":
            retain_undated(skipped, "spectral_lightcurve", data)
            continue
        data = subset(data, time_order(data["TIME"]))
        lc_ids = (ids[:, None] * 10 + np.arange(10)[None, :]).ravel()
        ctime = subset(lc, lc_ids)
        ctime = subset(ctime, time_order(ctime["TIME"]))
        spectrum = make_table(f"BTO{detector + 1}_CSPEC", data, HIST_SCHEMA)
        bands = energy_bands(detector, np.unique(data["BINMODE"]), calibration)
        lightcurve = make_table(f"BTO{detector + 1}_CTIME", ctime, CTIME_SCHEMA)
        # The shared observation covers both sets of end-stamped bins.
        coverage = np.concatenate((data["TIME"] - 1.0, data["TIME"],
                                   ctime["TIME"] - 0.1, ctime["TIME"]))
        hdus = [fits.PrimaryHDU(), spectrum, bands, lightcurve,
                observation_gti(coverage, "BTO_GTI")]
        for hdu in hdus:
            set_headers(hdu, coverage, detector, day, calibration)
        set_headers(spectrum, data["TIME"], detector, day, calibration, interval=1.0)
        # Restore extension classification after common metadata is installed.
        bands.header.update(HDUCLAS1="RESPONSE", HDUCLAS2="EBOUNDS")
        hdus[-1].header["HDUCLAS1"] = "GTI"
        spectrum.header.update(HDUCLAS2="EVRATE", TIMEDEL=1.0, TIMEPIXR=1.0,
                               DEADAPP=True, BINTYPE=bands.header["BINTYPE"], DTTICK=DEADTIME_TICK)
        set_headers(lightcurve, ctime["TIME"], detector, day, calibration, interval=0.1)
        lightcurve.header.update(HDUCLAS2="EVRATE", TIMEDEL=0.1, TIMEPIXR=1.0,
                                 DEADAPP=False, BINTYPE="DISCRIM", COUNTMTH="MODULO",
                                 CHAN1="ZERO_CROSSING", CHAN2="UPPER", CHAN3="SUPER_UPPER")
        lightcurve.header["COMMENT"] = "COUNT is minimum modulo increment; multiple counter wraps are unresolved."
        lightcurve.header["COMMENT"] = "No per-100ms deadtime is available. TIMEDEL is nominal 0.1 s."
        lightcurve.header["COMMENT"] = "COUNT=-1 where a preceding cumulative counter is unavailable."
        path = science_path(root, day, "bto", f"cs{day}bto{detector + 1}_spec_lc.fits")
        write_fits(path, hdus)
        products.append(dict(path=str(path.relative_to(root)), kind="spectral_lightcurve", detector=detector + 1,
                             source_ids=source_ids(data),
                             cspec_rows=len(ids), ctime_rows=len(ctime["TIME"]),
                             invalid_deadtime=int(np.sum(~np.isfinite(data["TIMEDEL"]))),
                             unknown_time=int(np.sum(~np.isfinite(data["TIME"])))))


def write_housekeeping(root, packets, calibration, products, skipped):
    table = decode_housekeeping(packets, calibration)
    first = [("TIME", "1D", "s", False), ("BTO_MODE", "1B", None, False),
             ("UPTIME_TICKS", "1J", None, True), ("LAST_GRB_TIME", "1D", "s", False),
             ("LAST_GRB_ID", "1J", None, True), ("ZERO_CROSSING_COUNT", "1I", None, True),
             ("SUPPER_UPPER_COUNT", "1I", None, True)]
    first += [(name, "1E", unit, False) for name, unit in SENSORS]
    for day, ids in date_groups(table["TIME"], table["PKT_TIME"]):
        data = subset(table, ids)
        if day == "undated":
            retain_undated(skipped, "housekeeping", data)
            continue
        data = subset(data, time_order(data["TIME"]))
        # One combined housekeeping extension, with the requested BTO1 label.
        # Original source detector IDs remain available in L0.
        hdu = make_table("BTO1_HSK", data, first)
        hdus = [fits.PrimaryHDU(), hdu]
        for item in hdus:
            set_headers(item, data["TIME"], None, day, calibration)
        present = np.unique(data["BTO_ID"])
        applied = [name in detector_calibration(calibration, int(det)).get("housekeeping", {})
                   for det in present for name, _ in SENSORS]
        status = "APPLIED" if all(applied) else "PARTIAL" if any(applied) else "UNAVAILABLE"
        hdu.header.update(HDUCLAS2="HKP", CALSTAT=status)
        for det in present:
            coefficients = detector_calibration(calibration, int(det)).get("housekeeping", {})
            for name, unit in SENSORS:
                if name in coefficients:
                    slope, intercept = coefficients[name]
                    hdu.header["HISTORY"] = (
                        f"BTO{det + 1} {name} = {slope:g}*ADC {intercept:+g} {unit}")
        hdu.header["COMMENT"] = "Combined BTO1/BTO2 housekeeping; original detector IDs and ADCs are in L0."
        hdu.header["COMMENT"] = "Missing sensor coefficients produce NaN; see CALSTAT and manifest.json."
        path = science_path(root, day, "auxil", f"cs{day}bto_gen_hk.fits")
        write_fits(path, hdus)
        products.append(dict(path=str(path.relative_to(root)), kind="housekeeping", rows=len(ids),
                             source_ids=source_ids(data)))


def write_events(root, packets, detector, calibration, products, skipped, registry):
    data, triggers, diagnostics = decode_events(packets)
    detcal = detector_calibration(calibration, detector)
    if "energy" in detcal:
        slope, intercept = detcal["energy"]
        data["ENERGY"] = (data["PHA"] * slope + intercept).astype(np.float32)
    groups = []
    for group in np.unique(data["GROUP"]):
        ids = np.flatnonzero(data["GROUP"] == group)
        valid = data["TIME"][ids]
        valid = valid[np.isfinite(valid)]
        fallback = data["PKT_TIME"][ids]
        fallback = fallback[np.isfinite(fallback)]
        route_time = float(valid.min()) if len(valid) else float(fallback.min()) if len(fallback) else np.nan
        groups.append((route_time, int(group), ids))
    groups.sort(key=lambda item: item[0] if np.isfinite(item[0]) else np.inf)
    known, used = registry
    for route_time, group, ids in groups:
        day = date_key(route_time)
        if day == "undated":
            retain_undated(skipped, "events", subset(data, ids))
            continue
        # Keep observation names stable when a later run selects other folders.
        # Existing observations keep their number; new ones use a free number.
        key = f"{day}:{int(triggers[group])}:{route_time.hex()}"
        if not np.any(np.isfinite(data["TIME"][ids])):
            key += f':{int(data["SOURCE_ID"][ids[0]])}:{int(data["BYTE_OFFSET"][ids[0]])}'
        if key in known:
            number = known[key]
        else:
            number = next((i for i in range(1, 1000) if i not in used[day]), None)
            if number is None:
                raise ParseError(f"More than 999 event groups on {day}; filename field exhausted")
            known[key] = number
            used[day].add(number)
        obs_id = f"{day}{number:03d}t"
        rows = subset(data, ids)
        rows = subset(rows, time_order(rows["TIME"]))
        hdu = make_table("BTO_EVENT", rows, EVENT_SCHEMA)
        hdus = [fits.PrimaryHDU(), hdu, observation_gti(rows["TIME"])]
        for item in hdus:
            set_headers(item, rows["TIME"], detector, obs_id, calibration)
            item.header["GRB_ID"] = (int(triggers[group]), "Onboard START ID; -1 if START is absent")
        hdu.header.update(HDUCLAS1="EVENT", HDUCLAS2="ALL", DEADAPP=False,
                          DTTICK=DEADTIME_TICK, DWTRES=DWT_QUANTUM / DWT_HZ,
                          TIMEMETH="ABSOLUTE_BRACKETS", CALSTAT="APPLIED" if "energy" in detcal else "UNAVAILABLE",
                          NTIMED=int(np.sum(np.isfinite(rows["TIME"]))),
                          NUNTIMED=int(np.sum(~np.isfinite(rows["TIME"]))),
                          ROUTETIM="EVENT" if np.any(np.isfinite(rows["TIME"])) else "PACKET")
        hdus[-1].header["HDUCLAS1"] = "GTI"
        if "energy" in detcal:
            slope, intercept = detcal["energy"]
            hdu.header.update(E_MIN=float(intercept), E_MAX=float(4095 * slope + intercept))
        hdu.header["COMMENT"] = "Original event words, flags and raw clocks are preserved in L0."
        hdu.header["COMMENT"] = "TIME is bracketed by absolute clocks; packet transmission time is not used."
        hdu.header["COMMENT"] = "FLAGS bits 0/1/2/3: pseudo/lower/upper/super-upper. PHA is unsigned 12-bit ADC."
        date_part = day[2:] if day != "undated" else day
        filename = f"cs{date_part}{number:03d}tbto{detector + 1}_evt.fits"
        path = science_path(root, day, "bto", filename)
        write_fits(path, hdus)
        products.append(dict(path=str(path.relative_to(root)), kind="events", detector=detector + 1,
                             source_ids=source_ids(rows),
                             event_key=key,
                             rows=len(ids), onboard_trigger_id=int(triggers[group]),
                             timed=hdu.header["NTIMED"], untimed=hdu.header["NUNTIMED"]))
    return diagnostics


# 9. Batch execution. Replace matching files; preserve all other archive files.
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded_path(path, output):
    stage_prefix = f".{output.name}-building-"
    return any(parent == output or
               (parent.parent == output.parent and parent.name.startswith(stage_prefix))
               for parent in (path, *path.parents))


def is_managed_archive(path):
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text()).get("archive_format") in (
            ARCHIVE_FORMAT, *PREVIOUS_ARCHIVE_FORMATS)
    except (OSError, ValueError, AttributeError):
        return False


def collect_inputs(targets, output, recursive=True):
    """Accept several files/folders or one parent folder; exclude output archives."""
    output = Path(output).expanduser().resolve()
    files = set()
    for target in targets:
        path = Path(target).expanduser().resolve()
        if is_excluded_path(path, output) or (path.is_dir() and is_managed_archive(path)):
            raise ParseError(f"An output archive cannot be used as an input: {path}")
        if path.is_file():
            if path.suffix.lower() != ".bin":
                raise ParseError(f"Expected a .bin file: {path}")
            files.add(path)
        elif path.is_dir():
            if recursive:
                for directory, dirs, names in os.walk(path, followlinks=False):
                    base = Path(directory)
                    dirs[:] = [name for name in dirs
                               if not is_excluded_path((base / name).resolve(), output)
                               and not is_managed_archive(base / name)]
                    files.update((base / name).resolve() for name in names
                                 if Path(name).suffix.lower() == ".bin"
                                 and not is_excluded_path((base / name).resolve(), output))
            else:
                files.update(p.resolve() for p in path.iterdir()
                             if p.is_file() and p.suffix.lower() == ".bin"
                             and not is_excluded_path(p.resolve(), output))
        else:
            raise ParseError(f"Input does not exist: {path}")
    if not files:
        raise ParseError("No .bin files found")
    return sorted(files)


def input_time_key(path, framing, checksum):
    """Folder names do not determine telemetry order across input captures."""
    for packet in read_packets(path, framing=framing, checksum=checksum):
        if packet.apid in SCIENCE_APIDS and np.isfinite(packet.time):
            return packet.time, path.name, str(path)
    return float("inf"), path.name, str(path)


def archive_member(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ParseError(f"Invalid relative archive path: {relative}")
    return root / path


def event_registry(output, detector, old_products):
    """Reserve existing filenames and recover observation identities on reruns."""
    known, used = {}, defaultdict(set)
    metadata = {p["path"]: p for p in old_products if p.get("kind") == "events"}
    pattern = re.compile(r"cs\d{6}(\d{3})tbto" + str(detector + 1) + r"_evt\.fits")
    for path in (output / "cosi/data/obs").glob(f"*/*/bto/*tbto{detector + 1}_evt.fits"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        day, number = path.parent.parent.name, int(match.group(1))
        used[day].add(number)
        entry = metadata.get(str(path.relative_to(output)), {})
        if "event_key" in entry:
            known[entry["event_key"]] = number
        else:
            # Compatibility with existing v5 products, which lacked event_key.
            with fits.open(path) as hdus:
                header = hdus[0].header
                if "TSTART" in header and "GRB_ID" in header:
                    key = f'{day}:{header["GRB_ID"]}:{float(header["TSTART"]).hex()}'
                    known[key] = number
                elif "DIAGFILE" in header and "GRB_ID" in header:
                    # Read an existing v5 companion only to preserve the name
                    # of an entirely untimed observation. No companion is written.
                    companion = archive_member(output, header["DIAGFILE"])
                    if companion.is_file():
                        with np.load(companion, allow_pickle=False) as raw:
                            packet_time = raw["EVENT__PKT_TIME"]
                            finite = packet_time[np.isfinite(packet_time)]
                            if len(finite):
                                key = f'{day}:{header["GRB_ID"]}:{float(finite.min()).hex()}'
                                key += f':{int(raw["EVENT__SOURCE_ID"][0])}:{int(raw["EVENT__BYTE_OFFSET"][0])}'
                                known[key] = number
    return known, used


def install_files(stage, output, clobber):
    """Preflight all collisions, then install per file with rollback on failure."""
    if not output.exists():
        stage.rename(output)
        return
    if not output.is_dir():
        raise ParseError("Output must be a directory")
    plan = []
    for source in sorted(stage.rglob("*"), key=lambda p: (p.name == "manifest.json", str(p))):
        if not source.is_file():
            continue
        relative = source.relative_to(stage)
        target = output / relative
        for part in (target, *target.parents):
            if part == output:
                break
            if part.is_symlink():
                raise ParseError(f"Refusing to replace through a symbolic link: {part}")
        if target.exists():
            if not target.is_file():
                raise ParseError(f"Output file conflicts with a directory: {target}")
            if relative.parts[0] == "L0" and sha256_file(source) == sha256_file(target):
                continue  # Identical original bytes need no rewrite, including clobber=no.
            if not clobber and relative != Path("manifest.json"):
                raise ParseError(f"Output file exists and clobber=no: {target}")
        plan.append((source, target, relative))
    installed = []
    try:
        for source, target, relative in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists():
                backup = stage / "_previous" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
            installed.append((target, backup))
            os.replace(source, target)
    except Exception:
        for target, backup in reversed(installed):
            if target.exists():
                target.unlink()
            if backup is not None:
                os.replace(backup, target)
        raise
    shutil.rmtree(stage)


def build_archive(paths, output, *, clobber=True, overwrite=None, framing="auto", checksum="absent",
                  calibration=None, level="l1"):
    if level not in ("l0", "l1", "all"):
        raise ParseError("Level must be l0, l1 or all; only unified L1 is supported")
    if overwrite is not None:  # Compatibility with callers of the older Python API.
        clobber = bool(overwrite)
    level = "l1" if level == "all" else level
    calibration = read_calibration() if calibration is None else validate_calibration(calibration)
    if Path(output).is_symlink():
        raise ParseError("Output directory must not be a symbolic link")
    output = Path(output).resolve()
    paths = sorted({Path(p).resolve() for p in paths})
    if not paths or any(is_excluded_path(p, output) for p in paths):
        raise ParseError("Provide input binaries outside the output archive")
    old = {}
    old_manifest = output / "manifest.json"
    if old_manifest.exists():
        if not is_managed_archive(output):
            raise ParseError("Existing manifest.json does not describe a BTO parser archive")
        old = json.loads(old_manifest.read_text())
    old_inputs = [dict(entry) for entry in old.get("inputs", [])]
    old_products = old.get("products", [])
    for entry in old_inputs:
        archive_member(output, entry["archived_path"])
    for entry in old_products:
        archive_member(output, entry["path"])
    paths.sort(key=lambda p: input_time_key(p, framing, checksum))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    report = dict(archive_format=ARCHIVE_FORMAT, parser_version=VERSION, inputs=old_inputs, products=[],
                  duplicates_skipped=[], packet_counts={}, sequence_gaps={}, photon_diagnostics={},
                  skipped_l1=[], selected_source_ids=[])
    report["settings"] = dict(framing=framing, checksum=checksum, level=level, clobber=bool(clobber),
                              deadtime_tick_seconds=DEADTIME_TICK, pps_hz=PPS_HZ, dwt_hz=DWT_HZ,
                              maximum_anchor_span_seconds=MAX_ANCHOR_SPAN,
                              clock_tolerance_ppm=CLOCK_TOLERANCE * 1e6)
    report["limitations"] = [
        "GTIs contain the observation start/stop in MET, not quality-screened intervals.",
        "A product with no finite scientific time has an unresolved, empty GTI.",
        "Records with no usable observation date stay in L0 and are listed under skipped_l1.",
        "CTIME uses nominal 0.1 s widths; per-bin deadtime is not in the packets.",
        "CTIME counts are minimum modulo increments; multiple wraps cannot be recovered.",
        "TEMP_EXT is uncalibrated unless its coefficients are explicitly supplied.",
        "Histogram channel/energy boundaries require the firmware's 30 ADC bin edges.",
        "HK units follow the ICD: C, V, mA.",
        "The two energy fits are at 17 C; no temperature correction, validity range or covariance was supplied.",
        "Only events bounded by consistent absolute clock markers are timed; unbounded edges remain NaN.",
        "CTIME COUNT uses 3J rather than 3I to retain uint16 values plus an unknown-count sentinel.",
        "Matching products are replaced from the selected inputs, not appended; select all inputs needed for each product.",
        "Packet counts and photon diagnostics describe this run; inputs and products also include retained older entries.",
    ]
    catalog = {entry["source_id"]: entry for entry in old_inputs}
    next_id = max(catalog, default=-1) + 1
    selected = set()
    streams = defaultdict(list)
    previous, seen_hashes = {}, {}
    counts, gaps = Counter(), Counter()
    try:
        for path in paths:
            digest = sha256_file(path)
            # Reuse an unchanged source's identity, including pre-v6 manifests.
            matches = [entry for entry in old_inputs
                       if entry["source_id"] not in selected and entry["sha256"] == digest
                       and entry["name"] == path.name
                       and entry.get("source_path", str(path)) == str(path)]
            if matches:
                entry = matches[0]
                source = entry["source_id"]
            else:
                while (output / "L0" / f"{next_id:04d}").exists():
                    next_id += 1
                source, next_id = next_id, next_id + 1
                entry = dict(source_id=source, name=path.name, sha256=digest,
                             archived_path=str(Path("L0") / f"{source:04d}" / path.name))
                report["inputs"].append(entry)
                catalog[source] = entry
            selected.add(source)
            report["selected_source_ids"].append(source)
            entry.update(source_path=str(path), bytes=path.stat().st_size)
            entry.pop("duplicate_of", None)
            copied = archive_member(stage, entry["archived_path"])
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copied)
            if sha256_file(copied) != digest:
                raise ParseError(f"Input changed while being copied: {path}")
            if digest in seen_hashes:
                original = seen_hashes[digest]
                entry.update(duplicate_of=original, packets=catalog[original]["packets"])
                report["duplicates_skipped"].append(str(path))
                continue
            seen_hashes[digest] = source
            print(f"Reading {path.name}", flush=True)
            source_packets = 0
            for packet in read_packets(copied, source, framing, checksum):
                source_packets += 1
                key = f"0x{packet.apid:03X}"
                counts[key] += 1
                if packet.apid not in SCIENCE_APIDS:
                    continue
                identity = packet.apid, packet.detector
                if identity in previous:
                    packet.gap = packet.sequence != (previous[identity] + 1) % 16384
                previous[identity] = packet.sequence
                if packet.gap:
                    gaps[f"{key}/BTO{packet.detector + 1}"] += 1
                if level != "l0":
                    streams[identity].append(packet)
            if not source_packets:
                raise ParseError(f"Empty telemetry file: {path}")
            entry["packets"] = source_packets
        report["packet_counts"], report["sequence_gaps"] = dict(counts), dict(gaps)
        for detector in (0, 1):
            packets = streams.pop((0xD6, detector), [])
            if packets:
                print(f"Writing BTO{detector + 1} spectral/lightcurve files", flush=True)
                write_histograms(stage, packets, detector, calibration, report["products"], report["skipped_l1"])
        hk = streams.pop((0xD8, 0), []) + streams.pop((0xD8, 1), [])
        if hk:
            print("Writing BTO1/BTO2 housekeeping files", flush=True)
            write_housekeeping(stage, hk, calibration, report["products"], report["skipped_l1"])
        del hk
        for detector in (0, 1):
            packets = streams.pop((0xD7, detector), [])
            if packets:
                print(f"Writing BTO{detector + 1} event files", flush=True)
                report["photon_diagnostics"][str(detector + 1)] = write_events(
                    stage, packets, detector, calibration, report["products"], report["skipped_l1"],
                    event_registry(output, detector, old_products))
        report["calibration"] = calibration
        report["written_products"] = [entry["path"] for entry in report["products"]]
        all_products = {entry["path"]: entry for entry in old_products
                        if archive_member(output, entry["path"]).is_file()}
        all_products.update({entry["path"]: entry for entry in report["products"]})
        report["products"] = list(all_products.values())
        report["retained_products"] = len(all_products) - len(report["written_products"])
        (stage / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        install_files(stage, output, clobber)
    except Exception:
        print(f"Run failed. Existing files were preserved; staged files are at {stage}.", file=sys.stderr)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Binary files or directories")
    parser.add_argument("--input", nargs="+", action="append", default=[],
                        help="One or more files/folders; may be repeated")
    parser.add_argument("-o", "--output", "--output-dir", dest="output_dir",
                        default="BTO_FITS_Archive", help="Archive directory")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Include input subfolders (default)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                        help="Read only the named folders, without subfolders")
    parser.add_argument("--clobber", choices=["yes", "no"], default="yes",
                        help="Overwrite matching output files (default: yes); preserve other files")
    parser.add_argument("--overwrite", dest="clobber", action="store_const", const="yes",
                        help="Alias for --clobber yes")
    parser.add_argument("--no-clobber", dest="clobber", action="store_const", const="no",
                        help="Alias for --clobber no")
    parser.add_argument("--framing", choices=["auto", "ccsds", "sync"], default="auto", help="auto detects optional EB90")
    parser.add_argument("--checksum", choices=["absent", "required"], default="absent", help="External BTO XOR16; absent in samples")
    parser.add_argument("--calibration", type=Path, help="JSON replacing the supplied energy/housekeeping defaults")
    parser.add_argument("--level", choices=["l0", "l1", "all"], default="l1",
                        help="l0 preserves binaries; l1/all also writes ICD L1 FITS")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args(argv)
    try:
        output_arg = Path(args.output_dir).expanduser()
        if output_arg.is_symlink():
            raise ParseError("Output directory must not be a symbolic link")
        output = output_arg.resolve()
        targets = args.files + [item for group in args.input for item in group]
        if not targets:
            parser.error("provide --input or positional .bin files")
        paths = collect_inputs(targets, output, args.recursive)
        report = build_archive(paths, output, clobber=args.clobber == "yes", framing=args.framing,
                               checksum=args.checksum, calibration=read_calibration(args.calibration), level=args.level)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(report['written_products'])} FITS files to {output}; "
          f"preserved {report['retained_products']} other indexed products")
    for detector, item in report["photon_diagnostics"].items():
        print(f"BTO{detector}: {item['events']:,} events; {item['timed_events']:,} with validated relative clock timing")
    print("See manifest.json for input hashes, packet counts, timing diagnostics and limitations.")
    if report["skipped_l1"]:
        print(f"Retained {sum(x['rows'] for x in report['skipped_l1'])} undated records in L0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
