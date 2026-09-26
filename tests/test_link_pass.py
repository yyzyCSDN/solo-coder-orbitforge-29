import math
import pytest
from fastapi.testclient import TestClient
from orbitforge.api.app import app
from orbitforge.link.atmospheric_loss import gaseous_loss_db, rain_loss_db
from orbitforge.link.budget import cn0_dbhz, free_space_loss_db, received_power_dbw
from orbitforge.link.doppler import doppler_shift_hz
from orbitforge.link.modulation import (MODCOD_ORDER, SwitchPolicy, bitrate_bps, required_ebn0,
                                        select_modcod_series)
from orbitforge.link.pass_budget import LinkConfig, PassSample, evaluate_pass

BW = 1e6
POLICY = SwitchPolicy(required_margin_db=1.0, hysteresis_db=1.0, min_dwell_s=30.0)

def margins_for_cn0(cn0):
    return {n: cn0 - 10 * math.log10(bitrate_bps(n, BW)) - required_ebn0(n) for n in MODCOD_ORDER}

def cn0_threshold(name, margin=0.0):
    return 10 * math.log10(bitrate_bps(name, BW)) + required_ebn0(name) + margin

def make_config(**kw):
    base = dict(freq_hz=2.2e9, tx_power_dbw=10.0, tx_gain_dbi=5.0, rx_gain_dbi=15.0,
                system_temp_k=300.0, bandwidth_hz=BW, fixed_losses_db=3.0,
                humidity=0.7, rain_rate_mm_h=5.0, policy=POLICY)
    base.update(kw)
    return LinkConfig(**base)

def synthetic_pass(duration=600.0, dt=10.0, range_rate=0.0):
    samples = []
    t = 0.0
    while t <= duration:
        frac = t / duration
        el = math.radians(5) + math.radians(75) * math.sin(math.pi * frac)
        rng = 5000.0 - 4500.0 * math.sin(math.pi * frac)
        samples.append(PassSample(t, rng, el, range_rate))
        t += dt
    return samples

def test_chain_matches_scalar_functions():
    cfg = make_config()
    s = PassSample(0.0, 1200.0, math.radians(40), -6.0)
    point = evaluate_pass([s], cfg).points[0]
    gas = gaseous_loss_db(cfg.freq_hz / 1e9, s.elevation_rad, cfg.humidity)
    rain = rain_loss_db(cfg.freq_hz / 1e9, cfg.rain_rate_mm_h, s.elevation_rad)
    pr = received_power_dbw(10.0, 5.0, 15.0, 1200.0, cfg.freq_hz, 3.0 + gas + rain)
    cn0 = cn0_dbhz(pr, 300.0)
    assert point.fspl_db == pytest.approx(free_space_loss_db(1200.0, cfg.freq_hz))
    assert point.gaseous_loss_db == pytest.approx(gas)
    assert point.rain_loss_db == pytest.approx(rain)
    assert point.doppler_hz == pytest.approx(doppler_shift_hz(cfg.freq_hz, -6.0))
    assert point.rx_power_dbw == pytest.approx(pr)
    assert point.cn0_dbhz == pytest.approx(cn0)
    assert point.modcod is not None
    rate = bitrate_bps(point.modcod, BW)
    assert point.bitrate_bps == pytest.approx(rate)
    assert point.ebn0_db == pytest.approx(cn0 - 10 * math.log10(rate))
    assert point.margin_db == pytest.approx(point.ebn0_db - required_ebn0(point.modcod))

def test_hysteresis_band_holds_current_modcod():
    # QPSK_1_2 margin sits inside [req, req+hyst): acquire once, never upgrade.
    times = [10.0 * i for i in range(12)]
    rows = [margins_for_cn0(cn0_threshold('QPSK_1_2', 1.5))] * len(times)
    choices, events = select_modcod_series(times, rows, POLICY)
    assert len(events) == 1 and events[0].reason == 'acquire'
    assert set(choices) == {'QPSK_1_2'}
    # Push QPSK_3_4 above the raised threshold, then let its margin collapse:
    # downgrade is immediate, and the link does not bounce back up.
    rows = ([margins_for_cn0(cn0_threshold('QPSK_1_2', 1.5))] * 6
            + [margins_for_cn0(cn0_threshold('QPSK_3_4', 2.5))] * 6
            + [margins_for_cn0(cn0_threshold('QPSK_1_2', 1.5))] * 6)
    times = [10.0 * i for i in range(len(rows))]
    choices, events = select_modcod_series(times, rows, POLICY)
    reasons = [(e.reason, e.from_modcod, e.to_modcod) for e in events]
    assert ('upgrade', 'QPSK_1_2', 'QPSK_3_4') in reasons
    assert ('downgrade', 'QPSK_3_4', 'QPSK_1_2') in reasons
    assert choices[-1] == 'QPSK_1_2'
    assert len(events) == 3

def test_min_dwell_blocks_upgrade_but_not_forced_downgrade():
    times = [10.0 * i for i in range(5)]
    rows = [margins_for_cn0(cn0_threshold('QPSK_1_2', 1.5)),
            margins_for_cn0(cn0_threshold('QPSK_3_4', 2.5)),
            margins_for_cn0(cn0_threshold('QPSK_3_4', 2.5)),
            margins_for_cn0(cn0_threshold('QPSK_3_4', 2.5)),
            margins_for_cn0(cn0_threshold('BPSK_1_2', 1.2))]
    choices, events = select_modcod_series(times, rows, POLICY)
    # upgrade at t=10 and t=20 blocked by dwell since acquire at t=0
    assert choices[1] == 'QPSK_1_2' and choices[2] == 'QPSK_1_2'
    assert choices[3] == 'QPSK_3_4'  # dwell expired at t=30
    # forced downgrade at t=40 happens immediately despite dwell
    assert choices[4] == 'BPSK_1_2'
    assert events[-1].reason == 'downgrade' and events[-1].t == 40.0

def test_no_flapping_around_threshold():
    # QPSK_1_2 margin oscillates across req but never reaches req+hyst
    times = [5.0 * i for i in range(80)]
    rows = [margins_for_cn0(cn0_threshold('QPSK_1_2', 1.0 - 0.7 * math.cos(t / 40.0 * 2 * math.pi)))
            for t in times]
    choices, events = select_modcod_series(times, rows, POLICY)
    assert set(choices) == {'BPSK_1_2'}
    assert len(events) == 1 and events[0].reason == 'acquire'

def test_degraded_band_between_zero_and_required_margin():
    times = [0.0, 10.0, 20.0, 30.0]
    rows = [margins_for_cn0(cn0_threshold('BPSK_1_2', m)) for m in (1.5, 0.5, -0.5, 1.5)]
    choices, events = select_modcod_series(times, rows, POLICY)
    assert choices == ['BPSK_1_2', 'BPSK_1_2', None, 'BPSK_1_2']
    assert [e.reason for e in events] == ['acquire', 'outage', 'acquire']

def test_outage_and_low_margin_windows_are_time_located():
    cfg = make_config()
    result = evaluate_pass(synthetic_pass(), cfg)
    assert len(result.points) == 61
    # weak link at the pass edges: one outage and one low-margin window on
    # the way in and on the way out, with concrete time bounds
    assert len(result.outage_windows) == 2
    assert len(result.low_margin_windows) == 2
    first_out, second_out = result.outage_windows
    assert first_out.start_t == 0.0
    assert second_out.end_t == 600.0
    for w in result.outage_windows:
        assert w.min_margin_db < 0.0
        assert w.start_t <= w.t_min_margin <= w.end_t
    for w in result.low_margin_windows:
        assert 0.0 <= w.min_margin_db < cfg.policy.required_margin_db
    # the degraded band is adjacent to the outage it leads into
    first_low = result.low_margin_windows[0]
    assert first_low.start_t == pytest.approx(first_out.end_t)
    assert first_low.end_t > first_low.start_t

def test_window_boundaries_interpolated_between_samples():
    cfg = make_config(rain_rate_mm_h=0.0, policy=SwitchPolicy(1.0, 1.0, 0.0))
    gas = gaseous_loss_db(cfg.freq_hz / 1e9, math.radians(90), cfg.humidity)
    # best margin decreases 0.6 dB per 10 s sample: crosses req=1 at t=33.33, 0 at t=50
    samples = []
    for i in range(11):
        cn0 = cn0_threshold('BPSK_1_2', 3.0 - 0.6 * i)
        pr = cn0 - 228.6 + 10 * math.log10(300.0)
        fspl = 10.0 + 5.0 + 15.0 - 3.0 - gas - pr
        rng = 10 ** ((fspl - 20 * math.log10(cfg.freq_hz) + 147.55221677811662) / 20) / 1000
        samples.append(PassSample(10.0 * i, rng, math.radians(90)))
    result = evaluate_pass(samples, cfg)
    assert len(result.low_margin_windows) == 1
    assert len(result.outage_windows) == 1
    low = result.low_margin_windows[0]
    out = result.outage_windows[0]
    assert low.start_t == pytest.approx(100 / 3, abs=1e-9)
    assert low.end_t == pytest.approx(50.0, abs=1e-9)
    assert out.start_t == pytest.approx(50.0, abs=1e-9)
    assert out.end_t == 100.0
    assert out.min_margin_db == pytest.approx(-3.0)
    assert out.t_min_margin == 100.0

def test_upgrades_respect_dwell_over_full_pass():
    result = evaluate_pass(synthetic_pass(), make_config())
    switches = result.switches
    for prev, cur in zip(switches, switches[1:]):
        if cur.reason == 'upgrade':
            assert cur.t - prev.t >= POLICY.min_dwell_s - 1e-9
    # smooth pass: a handful of clean rung transitions, not chatter
    assert len(result.switches) <= 12
    assert result.total_volume_mb > 0.0
    mid = result.points[len(result.points) // 2]
    assert mid.modcod == '16QAM_3_4'

def test_doppler_tracking_limit_forces_outage():
    fast = synthetic_pass(duration=200.0, dt=20.0, range_rate=8.0)
    free = evaluate_pass(fast, make_config())
    assert all(p.modcod is not None for p in free.points[2:-2])
    limited = evaluate_pass(fast, make_config(receiver_max_doppler_hz=50e3))
    assert all(p.modcod is None for p in limited.points)
    assert len(limited.outage_windows) == 1
    assert limited.outage_windows[0].start_t == 0.0
    assert limited.outage_windows[0].end_t == 200.0
    assert limited.total_volume_mb == 0.0
    expected = doppler_shift_hz(2.2e9, 8.0)
    assert abs(limited.points[0].doppler_hz - expected) < 1e-6
    assert abs(expected) > 50e3

def test_empty_pass():
    result = evaluate_pass([], make_config())
    assert result.points == () and result.total_volume_mb == 0.0

def test_api_link_pass_endpoint():
    c = TestClient(app)
    samples = [{'t': s.t, 'range_km': s.range_km, 'elevation_rad': s.elevation_rad,
                'range_rate_km_s': s.range_rate_km_s} for s in synthetic_pass(range_rate=8.0)]
    payload = dict(samples=samples, freq_hz=2.2e9, tx_power_dbw=10.0, tx_gain_dbi=5.0,
                   rx_gain_dbi=15.0, system_temp_k=300.0, bandwidth_hz=BW,
                   fixed_losses_db=3.0, humidity=0.7, rain_rate_mm_h=5.0,
                   receiver_max_doppler_hz=1.0)
    r = c.post('/v1/link/pass', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body['points']) == len(samples)
    assert body['outage_windows'] and body['outage_windows'][0]['kind'] == 'outage'
    # doppler-blocked samples carry -inf margin internally; JSON gets null
    assert body['points'][0]['best_margin_db'] is None
    assert body['total_volume_mb'] == 0.0
