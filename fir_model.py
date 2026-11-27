"""Frozen deterministic FIR oracle and area geometry for the two-lane receiver."""

from collections import Counter
from functools import lru_cache
import math

DOMAIN_NAMES = ("history_ch0", "history_ch1", "history_ch2", "history_ch3",
                "coefficients", "input_fifo", "input_replay", "accumulators",
                "fifo_pointers", "phase_history_head", "frame_identifiers",
                "mac_controller", "recovery_controller", "output_commit")


VARIABLES = (
    ("taps", (32, 64)), ("data_bits", (8, 10, 12, 16)),
    ("coefficient_bits", (8, 10, 12, 16)), ("accumulator_bits", (24, 32, 40)),
    ("rounding", ("truncate", "nearest")), ("overflow", ("wrap", "saturate")),
    ("fifo_frames", (16, 32, 64)), ("replay_frames", (32, 64, 128)),
    ("mac_lanes", (1, 2, 4)), ("recovery", ("stop", "flush", "replay")),
    ("correction", ("read_only", "writeback")),
)


BASELINE = (64, 16, 16, 32, "nearest", "saturate", 32, 128, 2, "replay", "writeback")

SINE = tuple(round(12000 * math.sin(2 * math.pi * n / 64)) for n in range(64))


@lru_cache(maxsize=2)
def coefficients(taps):
    center = (taps - 1) / 2
    values = []
    for n in range(taps):
        x = n - center
        sinc = .16 if x == 0 else math.sin(2 * math.pi * .08 * x) / (math.pi * x)
        values.append(sinc * (.54 - .46 * math.cos(2 * math.pi * n / (taps - 1))))
    scale = sum(values)
    result = [round(value / scale * 16384) for value in values]
    correction = (16384 - sum(result)) // 2
    result[taps // 2 - 1] += correction
    result[taps // 2] += correction
    return tuple(result)


def stimulus(frame, channel, seed):
    noise = ((frame * 1103515245 + channel * 12345 + seed * 69069) & 0xffff) % 601 - 300
    phase = (seed >> 8) & 63
    value = SINE[(frame * (channel + 1) + channel * 7 + phase) % 64] + SINE[(frame * 13 + channel * 3) % 64] // 4 + noise
    profile = (seed >> 16) & 3
    if profile == 1:
        value = value // 4 + (24000 if frame // 16 % 2 else -24000)
    elif profile == 2:
        value = value // 2 + (30000 if frame % 13 == seed % 13 else 0)
    elif profile == 3:
        value = value // 8 + (32767 if frame // 16 % 2 else -32768)
    return max(-32768, min(32767, value))


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def requantize(value, shift, nearest):
    if shift <= 0:
        return value << -shift
    return ((abs(value) + (1 << (shift - 1))) >> shift) * (-1 if value < 0 else 1) if nearest else value >> shift


def design(genome):
    genes = tuple(genome)
    choices = options({})
    if len(genes) != len(choices) or any(type(gene) is not int or gene not in allowed for gene, allowed in zip(genes, choices)):
        raise ValueError("FIR design must contain fourteen legal protection choices and eleven implementation choices")
    result = {name: values[genes[14 + index]] for index, (name, values) in enumerate(VARIABLES)}
    result["schemes"] = genes[:14]
    return result


def options(_config):
    return [tuple(range(4)) for _ in DOMAIN_NAMES] + [tuple(range(len(values))) for _, values in VARIABLES]


def reference(genome, frames, seed, *, ideal=False):
    """Independent time-domain convolution; no recovery state or RTL buffers."""
    return {frame: list(values) for frame, values in _reference_data(tuple(genome)[14:20], frames, seed, ideal)}


@lru_cache(maxsize=256)
def _reference_data(recipe, frames, seed, ideal):
    tail = tuple(values.index(value) for (_, values), value in zip(VARIABLES, BASELINE, strict=True))
    genome = (0,) * 14 + recipe + tail[6:]
    layout = design(genome)
    taps, data_bits, coefficient_bits = layout["taps"], layout["data_bits"], layout["coefficient_bits"]
    data_shift, coeff_shift = 16 - data_bits, 16 - coefficient_bits
    kernel = coefficients(taps)
    samples = [[stimulus(frame, channel, seed) for frame in range(frames)] for channel in range(4)]
    nearest = layout["rounding"] == "nearest"
    if not ideal:
        # Inputs and coefficients are fixed for the whole window. Preserve the
        # original tap order and accumulator wrap on every multiply-add.
        lower, upper = -(1 << (data_bits - 1)), (1 << (data_bits - 1)) - 1
        samples = [[max(lower, min(upper, requantize(value, data_shift, nearest)))
                    for value in channel] for channel in samples]
        quantized_kernel = [requantize(value, coeff_shift, nearest) for value in kernel]
    result = {}
    for frame in range(3, frames, 4):
        output = []
        for channel in range(4):
            if ideal:
                value = sum(samples[channel][frame - k] * kernel[k] for k in range(min(taps, frame + 1))) / 16384
                output.append(max(-32768, min(32767, value)))
                continue
            acc = 0
            for k in range(min(taps, frame + 1)):
                product = samples[channel][frame - k] * quantized_kernel[k]
                acc = signed(acc + product, layout["accumulator_bits"])
            shift = 14 - data_shift - coeff_shift
            acc = requantize(acc, shift, layout["rounding"] == "nearest")
            output.append(max(-32768, min(32767, acc)) if layout["overflow"] == "saturate" else signed(acc, 16))
        result[frame] = output
    return tuple((frame, tuple(values)) for frame, values in result.items())


def observe(row, genome, config, seed):
    expected = reference(genome, config["stream_frames"], seed)
    ideal = reference(genome, config["stream_frames"], seed, ideal=True)
    seen, last_id, duplicate, reordered, bad = set(), -1, 0, 0, 0
    errors, delivered, out_of_tolerance = [], {}, set()
    unexpected = 0
    for frame, cycle, *values in row["outputs"]:
        duplicate += frame in seen
        reordered += frame <= last_id
        seen.add(frame)
        last_id = frame
        if frame not in expected or values != expected[frame]:
            bad += 1
        unexpected += frame not in expected
        if frame in ideal:
            frame_errors = [abs(value - target) / 32768 for value, target in zip(values, ideal[frame])]
            errors.extend(frame_errors)
            if max(frame_errors) > config["fault_error_limit"]:
                out_of_tolerance.add(frame)
        if frame in expected and frame not in delivered:
            delivered[frame] = max(0, cycle - frame * config["input_period_cycles"])
    missing_ids = set(expected) - seen
    missing = len(missing_ids)
    burst = maximum_burst = 0
    for frame in expected:
        burst = burst + 1 if frame in out_of_tolerance or frame in missing_ids else 0
        maximum_burst = max(maximum_burst, burst)
    latency = list(delivered.values())
    deadline = sum(value > config["sample_deadline_cycles"] for value in latency) + missing
    latency.extend(row["cycles"] - frame * config["input_period_cycles"] for frame in missing_ids)
    tail = sorted(latency)[max(0, math.ceil(len(latency) * .99) - 1)] if latency else config["drain_cycles"]
    quality = sum(errors) / len(errors) if errors else 1.0
    resolution = max(1, math.ceil(row["cycles"] / 2048))
    histogram = Counter((value + resolution - 1) // resolution * resolution for value in latency)
    excess = row.get("unrecorded_output_frames", 0)
    layout = design(genome)
    complete_service = config["stream_frames"] // 4 * (4 + layout["taps"] // layout["mac_lanes"] + 2)
    service_cycles = row.get("service_cycles", complete_service)
    return {"failed": bool(missing or unexpected or out_of_tolerance or duplicate or reordered or deadline or row["overflow_frames"] or excess),
            "missing_output_frames": missing, "incorrect_output_frames": bad,
            "unexpected_output_frames": unexpected, "out_of_tolerance_frames": len(out_of_tolerance),
            "maximum_normalized_error": max(errors, default=1.),
            "maximum_error_burst_output_frames": maximum_burst,
            "duplicate_output_frames": duplicate, "reordered_output_frames": reordered,
            "unrecorded_output_frames": excess,
            "deadline_misses": deadline, "normalized_mae": quality, "p99_latency_cycles": tail,
            "latency_histogram": dict(histogram), "latency_resolution_cycles": resolution,
            "delivery_fraction": len(set(expected) & seen) / max(1, len(expected)),
            "sustained_rate": config["stream_frames"] * config["input_period_cycles"] / max(1, service_cycles),
            "service_cycles_per_input_frame": service_cycles / config["stream_frames"],
            "filter_group_delay_cycles": (layout["taps"] - 1) / 2 * config["input_period_cycles"],
            "output_samples": (len(row["outputs"]) + excess) * 4, **{key: value for key, value in row.items() if key != "outputs"}}


AXES = (
    *(f'history_ch{channel}_bank{bank}' for bank in range(2) for channel in range(4)),
    *(f'accumulator_ch{channel}' for channel in range(4)),
    'replay_payload', 'replay_frame_tag', 'commit_payload', 'commit_frame_valid', 'commit_watermark',
    'lanes', 'recovery', 'scrub', 'fifo_frames', 'replay_frames',
)


PROTECTION = 17


LANES, RECOVERY, SCRUB, FIFO, REPLAY = range(PROTECTION, len(AXES))


OPTIONS = ((0,1,2,3),)*PROTECTION + ((1,2,4),(0,1,2,3),(0,1),(16,32),(32,64))


@lru_cache(maxsize=128)
def _encoded_ge(scheme, payload, words=1):
    checks = max(2, next(r for r in range(8) if 2**r >= payload+r+1))
    redundancy = (0, 2*payload, checks+1, 1)[scheme]
    codec = redundancy*(3 if scheme == 2 else 1.5)
    return words*(5*(payload+redundancy)+codec)


def geometry(genome):
    """Return the canonical legal choices and the same counted-state GE proxy as FIR."""
    values = tuple(genome)
    if len(values) != len(OPTIONS) or any(type(v) is not int or v not in options for v, options in zip(values, OPTIONS)):
        return None
    if values[SCRUB] and 2 not in values[:PROTECTION]:
        values = (*values[:SCRUB], 0, *values[SCRUB+1:])
    lanes, fifo, replay = values[LANES], values[FIFO], values[REPLAY]
    history = sum(_encoded_ge(s, 16, 16) for s in values[:8])
    accumulators = sum(_encoded_ge(s, 32) for s in values[8:12])
    logs = (_encoded_ge(values[12], 16, 4*replay)+_encoded_ge(values[13],32,replay)
            +_encoded_ge(values[14],16,4)+_encoded_ge(values[15],32)+_encoded_ge(values[15],1)
            +_encoded_ge(values[16],32))
    # P3 holds coefficients, FIFO, queue pointers, phase, the uncommitted
    # frame ID, MAC controller and recovery-controller words in the receiver.
    fixed = ((16,32),(16,4*fifo),(32,fifo),(6,2),(7,1),(2,1),(6,1),(32,1),
             (7,1),(4,1),(6,1),(32,3),(7,1),(2,1),(1,1))
    area = (history+accumulators+logs+sum(_encoded_ge(3, width, count) for width,count in fixed)
            +4*lanes*16*16*2.4+4*lanes*32*5+32*16*.5+500)
    return values, round(area, 6)


def packed_hardware(genome):
    """Map genomic coordinates to the native fourteen legacy and ten bank registers."""
    canonical, area = geometry(genome)
    base = [3]*14
    base[:4] = canonical[:4]
    base[7], base[6], base[13] = canonical[8], canonical[12], canonical[14]
    extras = (*canonical[4:8], *canonical[9:12], canonical[13], canonical[15], canonical[16])
    settings = (*base, 0,3,3,1,1,1,(16,32,64).index(canonical[FIFO]),(32,64,128).index(canonical[REPLAY]),
                (1,2,4).index(canonical[LANES]),min(canonical[RECOVERY],2),canonical[SCRUB])
    return settings, sum(v << (3*i) for i,v in enumerate(extras))


def payload_position(bit, scheme):
    if scheme != 2:
        return bit
    # SECDED data go into non-parity Hamming positions, indexed from one.
    positions = [pos - 1 for pos in range(1, 72) if pos & (pos - 1)]
    return positions[bit]
