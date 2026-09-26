from __future__ import annotations
import math
from dataclasses import dataclass
from orbitforge.link.atmospheric_loss import gaseous_loss_db, rain_loss_db
from orbitforge.link.budget import cn0_dbhz, free_space_loss_db, received_power_dbw
from orbitforge.link.doppler import doppler_shift_hz
from orbitforge.link.modulation import MODCOD_ORDER, SwitchPolicy, ModcodSwitch, bitrate_bps, required_ebn0, select_modcod_series

@dataclass(frozen=True)
class PassSample:
    t: float
    range_km: float
    elevation_rad: float
    range_rate_km_s: float = 0.0

@dataclass(frozen=True)
class LinkConfig:
    freq_hz: float
    tx_power_dbw: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    system_temp_k: float
    bandwidth_hz: float
    fixed_losses_db: float = 0.0
    humidity: float = 0.5
    rain_rate_mm_h: float = 0.0
    receiver_max_doppler_hz: float | None = None
    policy: SwitchPolicy = SwitchPolicy()

@dataclass(frozen=True)
class BudgetPoint:
    t: float
    range_km: float
    elevation_rad: float
    fspl_db: float
    gaseous_loss_db: float
    rain_loss_db: float
    doppler_hz: float
    rx_power_dbw: float
    cn0_dbhz: float
    modcod: str | None
    bitrate_bps: float
    ebn0_db: float | None
    margin_db: float | None
    best_margin_db: float

@dataclass(frozen=True)
class MarginWindow:
    kind: str
    start_t: float
    end_t: float
    min_margin_db: float
    t_min_margin: float

@dataclass(frozen=True)
class PassResult:
    points: tuple[BudgetPoint, ...]
    switches: tuple[ModcodSwitch, ...]
    outage_windows: tuple[MarginWindow, ...]
    low_margin_windows: tuple[MarginWindow, ...]
    total_volume_mb: float

def _margin_row(cn0_dbhz, bandwidth_hz):
    return {name: cn0_dbhz - 10 * math.log10(bitrate_bps(name, bandwidth_hz)) - required_ebn0(name)
            for name in MODCOD_ORDER}

def _margin_windows(times, best_margin, required_margin_db):
    """Split the pass into ok / low_margin / outage runs.

    Run boundaries are the linearly interpolated crossings of the best
    achievable margin with the required-margin and zero-margin levels, so
    degraded and outage periods are pinned to concrete time intervals.
    """
    thresholds = (required_margin_db, 0.0)
    pts_t, pts_b = [times[0]], [best_margin[0]]
    for i in range(1, len(times)):
        t0, b0 = times[i - 1], best_margin[i - 1]
        t1, b1 = times[i], best_margin[i]
        crossings = []
        for level in thresholds:
            d0, d1 = b0 - level, b1 - level
            if d0 * d1 < 0.0:
                crossings.append((-d0 / (b1 - b0), level))
        for frac, level in sorted(crossings):
            pts_t.append(t0 + frac * (t1 - t0))
            pts_b.append(level)
        pts_t.append(t1)
        pts_b.append(b1)

    def state(b):
        if b < 0.0:
            return 'outage'
        return 'low_margin' if b < required_margin_db else 'ok'

    windows, prev_kind = [], 'ok'
    for j in range(len(pts_t) - 1):
        kind = state(0.5 * (pts_b[j] + pts_b[j + 1]))
        if kind == 'ok':
            prev_kind = 'ok'
            continue
        if prev_kind == kind and windows and windows[-1]['end_t'] == pts_t[j]:
            windows[-1]['end_t'] = pts_t[j + 1]
        else:
            windows.append({'kind': kind, 'start_t': pts_t[j], 'end_t': pts_t[j + 1], 'samples': []})
        windows[-1]['samples'].append((pts_b[j], pts_t[j]))
        windows[-1]['samples'].append((pts_b[j + 1], pts_t[j + 1]))
        prev_kind = kind
    if len(pts_t) == 1 and state(pts_b[0]) != 'ok':
        windows.append({'kind': state(pts_b[0]), 'start_t': pts_t[0], 'end_t': pts_t[0],
                        'samples': [(pts_b[0], pts_t[0])]})
    out = []
    for w in windows:
        min_b, t_min = min(w['samples'])
        out.append(MarginWindow(w['kind'], w['start_t'], w['end_t'], min_b, t_min))
    return out

def evaluate_pass(samples, config):
    """Chain FSPL, weather, Doppler and adaptive modcod over a pass."""
    if not samples:
        return PassResult((), (), (), (), 0.0)
    freq_ghz = config.freq_hz / 1e9
    times, rows, chain = [], [], []
    for s in samples:
        fspl = free_space_loss_db(s.range_km, config.freq_hz)
        gas = gaseous_loss_db(freq_ghz, s.elevation_rad, config.humidity)
        rain = rain_loss_db(freq_ghz, config.rain_rate_mm_h, s.elevation_rad) if config.rain_rate_mm_h > 0.0 else 0.0
        doppler = doppler_shift_hz(config.freq_hz, s.range_rate_km_s)
        pr = received_power_dbw(config.tx_power_dbw, config.tx_gain_dbi, config.rx_gain_dbi,
                                s.range_km, config.freq_hz, config.fixed_losses_db + gas + rain)
        cn0 = cn0_dbhz(pr, config.system_temp_k)
        row = _margin_row(cn0, config.bandwidth_hz)
        if config.receiver_max_doppler_hz is not None and abs(doppler) > config.receiver_max_doppler_hz:
            row = {name: float('-inf') for name in row}
        times.append(s.t)
        rows.append(row)
        chain.append((s, fspl, gas, rain, doppler, pr, cn0))

    choices, events = select_modcod_series(times, rows, config.policy)
    points = []
    for (s, fspl, gas, rain, doppler, pr, cn0), row, modcod in zip(chain, rows, choices):
        if modcod is None:
            points.append(BudgetPoint(s.t, s.range_km, s.elevation_rad, fspl, gas, rain, doppler,
                                      pr, cn0, None, 0.0, None, None, max(row.values())))
        else:
            rate = bitrate_bps(modcod, config.bandwidth_hz)
            ebn0 = cn0 - 10 * math.log10(rate)
            points.append(BudgetPoint(s.t, s.range_km, s.elevation_rad, fspl, gas, rain, doppler,
                                      pr, cn0, modcod, rate, ebn0, row[modcod], max(row.values())))

    best = [p.best_margin_db for p in points]
    windows = _margin_windows(times, best, config.policy.required_margin_db)
    volume_bits = 0.0
    for a, b in zip(points, points[1:]):
        volume_bits += max(0.0, b.t - a.t) * 0.5 * (a.bitrate_bps + b.bitrate_bps)
    return PassResult(tuple(points), tuple(events),
                      tuple(w for w in windows if w.kind == 'outage'),
                      tuple(w for w in windows if w.kind == 'low_margin'),
                      volume_bits / 8e6)
