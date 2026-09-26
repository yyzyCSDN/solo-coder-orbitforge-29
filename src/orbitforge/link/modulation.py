from __future__ import annotations
from dataclasses import dataclass
MODCOD = {'BPSK_1_2': (1, 0.5, 2.0), 'QPSK_1_2': (2, 0.5, 2.5), 'QPSK_3_4': (2, 0.75, 4.5), '8PSK_2_3': (3, 2 / 3, 7.5), '16QAM_3_4': (4, 0.75, 10.5)}
MODCOD_ORDER = tuple(sorted(MODCOD, key=lambda n: MODCOD[n][0] * MODCOD[n][1]))

def spectral_efficiency(name):
    bits, rate, _ = MODCOD[name]
    return bits * rate

def required_ebn0(name):
    return MODCOD[name][2]

def bitrate_bps(name, bandwidth_hz):
    return spectral_efficiency(name) * bandwidth_hz

def choose_modcod(ebn0_db, margin_db=1.0):
    viable = [(spectral_efficiency(n), n) for n in MODCOD if ebn0_db >= required_ebn0(n) + margin_db]
    return max(viable)[1] if viable else None

@dataclass(frozen=True)
class SwitchPolicy:
    required_margin_db: float = 1.0
    hysteresis_db: float = 1.0
    min_dwell_s: float = 30.0

@dataclass(frozen=True)
class ModcodSwitch:
    t: float
    from_modcod: str | None
    to_modcod: str | None
    reason: str

def _switch_reason(old, new):
    if old is None:
        return 'acquire'
    if new is None:
        return 'outage'
    return 'upgrade' if MODCOD_ORDER.index(new) > MODCOD_ORDER.index(old) else 'downgrade'

def select_modcod_series(times, margins, policy=SwitchPolicy()):
    """Pick a modcod per sample from per-sample margin rows.

    ``margins[i]`` maps modcod name to margin in dB above that modcod's
    required Eb/N0. Invariants: every upward move (including recovery from
    a degraded state) requires margin >= required + hysteresis and a minimum
    dwell since the last switch; downward moves forced by link closure are
    immediate. When no modcod meets the required margin the most robust one
    that still closes (margin >= 0) is used and flagged degraded via its
    margin value; ``None`` means outage. Returns (choices, events).
    """
    if len(times) != len(margins):
        raise ValueError('times/margins length mismatch')
    req = policy.required_margin_db
    up = req + policy.hysteresis_db

    def highest_with(row, threshold):
        viable = [n for n in MODCOD_ORDER if row[n] >= threshold]
        return viable[-1] if viable else None

    choices, events = [], []
    current, last_switch_t = None, None
    for t, row in zip(times, margins):
        target = current
        if current is not None and row[current] >= req:
            candidate = highest_with(row, up)
            if (candidate is not None and MODCOD_ORDER.index(candidate) > MODCOD_ORDER.index(current)
                    and (last_switch_t is None or t - last_switch_t >= policy.min_dwell_s)):
                target = candidate
        else:
            target = highest_with(row, req)
            if current is not None and target is not None and MODCOD_ORDER.index(target) > MODCOD_ORDER.index(current):
                # recovery from a degraded state is an upgrade: hysteresis + dwell apply
                recovered = highest_with(row, up)
                dwell_ok = last_switch_t is None or t - last_switch_t >= policy.min_dwell_s
                target = recovered if (recovered is not None
                                       and MODCOD_ORDER.index(recovered) > MODCOD_ORDER.index(current)
                                       and dwell_ok) else current
            if target is None:
                robust = [n for n in MODCOD_ORDER if row[n] >= 0.0]
                target = robust[0] if robust else None
        if target != current:
            events.append(ModcodSwitch(t, current, target, _switch_reason(current, target)))
            current, last_switch_t = target, t
        choices.append(current)
    return choices, events
