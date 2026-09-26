"""Time-varying link budget over one overpass.

Chains the point models (doppler -> observed frequency -> free-space,
gaseous and rain loss -> received power -> C/N0 -> per-modcod margin)
into a pass curve, selects the modcod adaptively with hysteresis and a
minimum dwell time, and reports outage / margin-deficit intervals
instead of only pass-averaged figures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from orbitforge.link.atmospheric_loss import gaseous_loss_db, rain_loss_db
from orbitforge.link.budget import cn0_dbhz, free_space_loss_db, received_power_dbw
from orbitforge.link.doppler import doppler_shift_hz
from orbitforge.link.modulation import MODCOD, MODCOD_ORDER, bitrate_bps, required_ebn0, spectral_efficiency

STATUS_OK = 'ok'
STATUS_DEFICIT = 'deficit'
STATUS_OUTAGE = 'outage'


@dataclass(frozen=True)
class PassGeometry:
    t_s: Sequence[float]
    range_km: Sequence[float]
    elevation_rad: Sequence[float]
    range_rate_km_s: Sequence[float]

    def __post_init__(self):
        n = len(self.t_s)
        if n == 0:
            raise ValueError('empty pass geometry')
        for seq in (self.range_km, self.elevation_rad, self.range_rate_km_s):
            if len(seq) != n:
                raise ValueError('geometry length mismatch')
        for a, b in zip(self.t_s, self.t_s[1:]):
            if b <= a:
                raise ValueError('t_s must be strictly increasing')


@dataclass(frozen=True)
class LinkBudgetConfig:
    freq_hz: float
    tx_power_dbw: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    system_temp_k: float
    symbol_rate_sps: float
    other_losses_db: float = 0.0
    humidity: float = 0.5
    rain_rate_mm_h: float | Sequence[float] = 0.0
    implementation_margin_db: float = 1.0
    hysteresis_db: float = 2.0
    min_dwell_s: float = 30.0
    modcods: Sequence[str] | None = None

    def __post_init__(self):
        if self.freq_hz <= 0:
            raise ValueError('freq_hz must be positive')
        if self.symbol_rate_sps <= 0:
            raise ValueError('symbol_rate_sps must be positive')
        if self.system_temp_k <= 0:
            raise ValueError('system_temp_k must be positive')
        if self.implementation_margin_db < 0:
            raise ValueError('implementation_margin_db must be >= 0')
        if self.hysteresis_db < 0:
            raise ValueError('hysteresis_db must be >= 0')
        if self.min_dwell_s < 0:
            raise ValueError('min_dwell_s must be >= 0')
        if self.modcods is not None:
            if len(self.modcods) == 0:
                raise ValueError('modcods must not be empty')
            unknown = [m for m in self.modcods if m not in MODCOD]
            if unknown:
                raise ValueError(f'unknown modcods: {unknown}')


@dataclass(frozen=True)
class PassSample:
    t_s: float
    elevation_rad: float
    range_km: float
    doppler_hz: float
    rx_frequency_hz: float
    fspl_db: float
    gas_loss_db: float
    rain_loss_db: float
    rx_power_dbw: float
    cn0_dbhz: float
    ebn0_db: float
    modcod: str | None
    bitrate_bps: float
    margin_db: float
    status: str


@dataclass(frozen=True)
class ModcodSwitch:
    t_s: float
    from_modcod: str | None
    to_modcod: str | None
    reason: str


@dataclass(frozen=True)
class LinkInterval:
    start_s: float
    end_s: float
    min_margin_db: float

    @property
    def duration_s(self):
        return self.end_s - self.start_s


@dataclass(frozen=True)
class PassBudgetResult:
    samples: tuple
    switches: tuple
    outage_intervals: tuple
    margin_deficit_intervals: tuple
    total_volume_mb: float
    availability: float
    min_margin_db: float
    max_doppler_hz: float

    @property
    def pass_duration_s(self):
        return self.samples[-1].t_s - self.samples[0].t_s

    @property
    def outage_time_s(self):
        return sum(iv.duration_s for iv in self.outage_intervals)

    @property
    def deficit_time_s(self):
        return sum(iv.duration_s for iv in self.margin_deficit_intervals)


def select_modcod_series(margins_db, t_s, implementation_margin_db=1.0, hysteresis_db=2.0, min_dwell_s=30.0):
    """Adaptive modcod selection with hysteresis and minimum dwell time.

    margins_db[i][k] is the margin (dB over required Eb/N0) of level k at
    t_s[i]; levels are ordered by increasing spectral efficiency. Returns
    (states, switches): states[i] is the selected level index or None for
    outage; switches are (t_s, from_level, to_level, reason) tuples.

    Transition rules:
    - (re)acquire from outage: highest level with margin >= hysteresis_db
    - safety downshift (margin < 0): highest level with margin >= 0, else
      outage; applied immediately, dwell does not block it
    - margin downshift (margin < implementation_margin_db): highest lower
      level with margin >= implementation_margin_db, if any; immediate
    - upshift: highest level with margin >= implementation_margin_db +
      hysteresis_db, only after min_dwell_s since the last switch
    """
    n = len(t_s)
    if n == 0:
        return [], []
    if len(margins_db) != n:
        raise ValueError('margins/t length mismatch')
    n_levels = len(margins_db[0])
    if n_levels == 0:
        raise ValueError('no modcod levels')
    if any(len(m) != n_levels for m in margins_db):
        raise ValueError('ragged margins')

    states = []
    switches = []
    state = None
    t_last_switch = None
    for i in range(n):
        t = t_s[i]
        m = margins_db[i]

        def highest(pred):
            best = None
            for k in range(n_levels):
                if pred(k):
                    best = k
            return best

        if state is None:
            target = highest(lambda k: m[k] >= hysteresis_db)
            if target is not None:
                switches.append((t, None, target, 'acquire' if not switches else 'reacquire'))
                state = target
                t_last_switch = t
        elif m[state] < 0.0:
            target = highest(lambda k: m[k] >= 0.0)
            switches.append((t, state, target, 'downshift_safety' if target is not None else 'outage'))
            state = target
            t_last_switch = t
        elif m[state] < implementation_margin_db:
            target = highest(lambda k: k < state and m[k] >= implementation_margin_db)
            if target is not None:
                switches.append((t, state, target, 'downshift_margin'))
                state = target
                t_last_switch = t
        else:
            target = highest(lambda k: m[k] >= implementation_margin_db + hysteresis_db)
            if target is not None and target > state and t - t_last_switch >= min_dwell_s:
                switches.append((t, state, target, 'upshift'))
                state = target
                t_last_switch = t
        states.append(state)
    return states, switches


def run_pass_budget(geometry: PassGeometry, config: LinkBudgetConfig) -> PassBudgetResult:
    levels = tuple(sorted(config.modcods, key=spectral_efficiency)) if config.modcods else MODCOD_ORDER
    t_s = list(geometry.t_s)
    n = len(t_s)
    rain = _broadcast_rain(config.rain_rate_mm_h, n)
    bitrates = [bitrate_bps(m, config.symbol_rate_sps) for m in levels]
    required = [required_ebn0(m) for m in levels]
    log_bitrates = [10.0 * math.log10(b) for b in bitrates]

    cn0s = []
    margins = []
    raw = []
    for i in range(n):
        el = geometry.elevation_rad[i]
        rng = geometry.range_km[i]
        dop = doppler_shift_hz(config.freq_hz, geometry.range_rate_km_s[i])
        f_obs = config.freq_hz + dop
        fspl = free_space_loss_db(rng, f_obs)
        gas = gaseous_loss_db(f_obs * 1e-9, el, config.humidity)
        rn = rain_loss_db(f_obs * 1e-9, rain[i], el) if rain[i] > 0.0 else 0.0
        pr = received_power_dbw(config.tx_power_dbw, config.tx_gain_dbi, config.rx_gain_dbi,
                                rng, f_obs, config.other_losses_db + gas + rn)
        cn0 = cn0_dbhz(pr, config.system_temp_k)
        cn0s.append(cn0)
        margins.append([cn0 - lb - rq for lb, rq in zip(log_bitrates, required)])
        raw.append((dop, f_obs, fspl, gas, rn, pr))

    states, raw_switches = select_modcod_series(
        margins, t_s, config.implementation_margin_db, config.hysteresis_db, config.min_dwell_s)
    switches = tuple(
        ModcodSwitch(t, levels[a] if a is not None else None, levels[b] if b is not None else None, reason)
        for t, a, b, reason in raw_switches)

    statuses = []
    sample_margins = []
    samples = []
    for i in range(n):
        k = states[i]
        sel = k if k is not None else 0
        ebn0 = cn0s[i] - log_bitrates[sel]
        margin = ebn0 - required[sel]
        if k is None:
            status = STATUS_OUTAGE
        elif margin < config.implementation_margin_db:
            status = STATUS_DEFICIT
        else:
            status = STATUS_OK
        statuses.append(status)
        sample_margins.append(margin)
        dop, f_obs, fspl, gas, rn, pr = raw[i]
        samples.append(PassSample(
            t_s=t_s[i], elevation_rad=geometry.elevation_rad[i], range_km=geometry.range_km[i],
            doppler_hz=dop, rx_frequency_hz=f_obs, fspl_db=fspl, gas_loss_db=gas, rain_loss_db=rn,
            rx_power_dbw=pr, cn0_dbhz=cn0s[i], ebn0_db=ebn0,
            modcod=levels[k] if k is not None else None,
            bitrate_bps=bitrates[k] if k is not None else 0.0,
            margin_db=margin, status=status))

    outage_intervals = _extract_intervals(t_s, margins, states, statuses, sample_margins, STATUS_OUTAGE,
                                          config.implementation_margin_db, config.hysteresis_db)
    deficit_intervals = _extract_intervals(t_s, margins, states, statuses, sample_margins, STATUS_DEFICIT,
                                           config.implementation_margin_db, config.hysteresis_db)

    span = t_s[-1] - t_s[0]
    outage_time = sum(iv.duration_s for iv in outage_intervals)
    availability = 1.0 - outage_time / span if span > 0 else (0.0 if outage_intervals else 1.0)
    return PassBudgetResult(
        samples=tuple(samples),
        switches=switches,
        outage_intervals=outage_intervals,
        margin_deficit_intervals=deficit_intervals,
        total_volume_mb=_integrate_volume_mb(t_s, [s.bitrate_bps for s in samples]),
        availability=availability,
        min_margin_db=min(sample_margins),
        max_doppler_hz=max(abs(s.doppler_hz) for s in samples))


def _broadcast_rain(value, n):
    if isinstance(value, (int, float)):
        values = [float(value)] * n
    else:
        values = [float(v) for v in value]
        if len(values) != n:
            raise ValueError('rain_rate_mm_h length must match pass samples')
    if any(v < 0.0 for v in values):
        raise ValueError('rain_rate_mm_h must be >= 0')
    return values


def _integrate_volume_mb(t_s, bitrates_bps):
    bits = 0.0
    for i in range(len(t_s) - 1):
        bits += (t_s[i + 1] - t_s[i]) * 0.5 * (bitrates_bps[i] + bitrates_bps[i + 1])
    return bits / 8000000.0


def _cross_time(t0, t1, y0, y1, threshold):
    d = y1 - y0
    if d == 0.0:
        return t1
    f = (threshold - y0) / d
    return t0 + min(1.0, max(0.0, f)) * (t1 - t0)


def _boundary(states, statuses, left, right, implementation_margin_db, hysteresis_db):
    sl, sr = statuses[left], statuses[right]
    if sr == STATUS_OUTAGE:
        return 0, 0.0
    if sl == STATUS_OUTAGE:
        return 0, hysteresis_db
    if sl == STATUS_OK and sr == STATUS_DEFICIT:
        lvl = states[right] if states[right] == states[left] else states[left]
        return lvl, implementation_margin_db
    if sl == STATUS_DEFICIT and sr == STATUS_OK:
        return states[right], implementation_margin_db
    return None


def _extract_intervals(t_s, margins, states, statuses, sample_margins, want,
                       implementation_margin_db, hysteresis_db):
    n = len(t_s)
    intervals = []
    i = 0
    while i < n:
        if statuses[i] != want:
            i += 1
            continue
        j = i
        while j + 1 < n and statuses[j + 1] == want:
            j += 1
        start = t_s[i]
        if i > 0:
            b = _boundary(states, statuses, i - 1, i, implementation_margin_db, hysteresis_db)
            if b is not None:
                lvl, thr = b
                start = _cross_time(t_s[i - 1], t_s[i], margins[i - 1][lvl], margins[i][lvl], thr)
        end = t_s[j]
        if j + 1 < n:
            b = _boundary(states, statuses, j, j + 1, implementation_margin_db, hysteresis_db)
            if b is not None:
                lvl, thr = b
                end = _cross_time(t_s[j], t_s[j + 1], margins[j][lvl], margins[j + 1][lvl], thr)
        intervals.append(LinkInterval(start, end, min(sample_margins[i:j + 1])))
        i = j + 1
    return tuple(intervals)
