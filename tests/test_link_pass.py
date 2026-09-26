import math
import dataclasses
from orbitforge.link.modulation import required_ebn0
from orbitforge.link.pass_budget import (
    LinkBudgetConfig, PassGeometry, run_pass_budget, select_modcod_series,
    STATUS_DEFICIT, STATUS_OK, STATUS_OUTAGE)


def synthetic_pass(n=301, duration_s=900.0, alt_km=600.0, cross_km=700.0, speed_km_s=7.5):
    t_s, rng, el, rr = [], [], [], []
    for i in range(n):
        t = duration_s * i / (n - 1)
        y = speed_km_s * (t - duration_s / 2)
        r = math.sqrt(cross_km ** 2 + y * y + alt_km ** 2)
        t_s.append(t)
        rng.append(r)
        el.append(math.asin(alt_km / r))
        rr.append(speed_km_s * y / r)
    return PassGeometry(t_s, rng, el, rr)


def base_config(**kw):
    return LinkBudgetConfig(freq_hz=12e9, tx_power_dbw=0.0, tx_gain_dbi=12.0, rx_gain_dbi=25.0,
                            system_temp_k=500.0, symbol_rate_sps=3e7, **kw)


def config_with_tca_cn0(geom, target_cn0, **kw):
    cfg = base_config(**kw)
    res = run_pass_budget(geom, cfg)
    tca = max(res.samples, key=lambda s: s.elevation_rad)
    return dataclasses.replace(cfg, tx_power_dbw=cfg.tx_power_dbw + (target_cn0 - tca.cn0_dbhz))


def test_pass_budget_chain_is_time_varying():
    geom = synthetic_pass()
    res = run_pass_budget(geom, config_with_tca_cn0(geom, 94.0))
    assert len(res.samples) == len(geom.t_s)
    mid = res.samples[len(res.samples) // 2]
    assert mid.fspl_db == min(s.fspl_db for s in res.samples)
    best = max(res.samples, key=lambda s: s.cn0_dbhz)
    assert abs(best.t_s - mid.t_s) < 1e-9
    cn0s = [s.cn0_dbhz for s in res.samples]
    assert max(cn0s) - min(cn0s) > 5.0
    for s in res.samples:
        if s.modcod is not None:
            assert abs(s.margin_db - (s.ebn0_db - required_ebn0(s.modcod))) < 1e-9
    assert res.samples[0].doppler_hz > 0 > res.samples[-1].doppler_hz
    assert abs(res.samples[0].rx_frequency_hz - (12e9 + res.samples[0].doppler_hz)) < 1e-6
    assert res.max_doppler_hz == max(abs(s.doppler_hz) for s in res.samples)


def test_adaptive_modcod_tracks_pass_curve():
    geom = synthetic_pass()
    res = run_pass_budget(geom, config_with_tca_cn0(geom, 94.0))
    tca = max(res.samples, key=lambda s: s.elevation_rad)
    assert tca.modcod == '16QAM_3_4'
    assert res.samples[0].modcod in ('BPSK_1_2', 'QPSK_1_2', 'QPSK_3_4')
    reasons = {sw.reason for sw in res.switches}
    assert 'upshift' in reasons and 'downshift_margin' in reasons
    upshifts = [sw for sw in res.switches if sw.reason == 'upshift']
    assert len(upshifts) >= 2
    assert all(sw.t_s < tca.t_s for sw in upshifts)
    for prev, cur in zip(res.switches, res.switches[1:]):
        if cur.reason == 'upshift':
            assert cur.t_s - prev.t_s >= 30.0 - 1e-9


def test_hysteresis_stops_threshold_flapping():
    t = [float(i) for i in range(400)]
    margins = [[6.0, 1.0 + 0.5 * math.sin(2.0 * math.pi * ti / 10.0)] for ti in t]
    calm_states, calm_sw = select_modcod_series(margins, t, 1.0, 2.0, 30.0)
    assert len(calm_sw) <= 1
    assert all(s == 0 for s in calm_states)
    _, twitchy_sw = select_modcod_series(margins, t, 1.0, 0.0, 0.0)
    assert len(twitchy_sw) >= 20


def test_min_dwell_delays_upshift():
    t = [float(i) for i in range(100)]

    def m1(ti):
        if ti < 30:
            return -5.0
        if ti < 40:
            return 5.0
        if ti < 50:
            return -1.0
        return 5.0

    margins = [[6.0, m1(ti)] for ti in t]
    _, sw = select_modcod_series(margins, t, 1.0, 2.0, 15.0)
    assert [s[0] for s in sw if s[3] == 'upshift'] == [30.0, 55.0]
    assert any(s[3] == 'downshift_safety' and s[0] == 40.0 for s in sw)
    _, sw0 = select_modcod_series(margins, t, 1.0, 2.0, 0.0)
    assert [s[0] for s in sw0 if s[3] == 'upshift'] == [30.0, 50.0]


def test_outage_and_reacquire_hysteresis():
    t = [float(i) for i in range(60)]

    def m0(ti):
        if ti < 10:
            return 5.0
        if ti < 20:
            return -2.0
        if ti < 30:
            return 1.0
        return 3.0

    margins = [[m0(ti)] for ti in t]
    states, sw = select_modcod_series(margins, t, 1.0, 2.0, 0.0)
    assert states[0] == 0 and states[15] is None and states[25] is None and states[35] == 0
    assert [s[3] for s in sw] == ['acquire', 'outage', 'reacquire']


def test_outage_and_deficit_intervals_are_time_located():
    geom = synthetic_pass()
    cfg = config_with_tca_cn0(geom, 76.0, implementation_margin_db=2.0, hysteresis_db=0.5)
    res = run_pass_budget(geom, cfg)
    assert len(res.outage_intervals) == 2
    assert len(res.margin_deficit_intervals) == 2
    first_out, second_out = res.outage_intervals
    t0, t1 = res.samples[0].t_s, res.samples[-1].t_s
    assert first_out.start_s == t0
    assert first_out.end_s < second_out.start_s
    assert second_out.end_s == t1
    for d in res.margin_deficit_intervals:
        assert first_out.end_s <= d.start_s and d.end_s <= second_out.start_s
        assert d.duration_s > 0.0
        for o in res.outage_intervals:
            assert d.end_s <= o.start_s or d.start_s >= o.end_s
    for s in res.samples:
        if s.status == STATUS_DEFICIT:
            assert s.modcod is not None and 0.0 <= s.margin_db < 2.0
        if s.status == STATUS_OUTAGE:
            assert s.modcod is None and s.margin_db < 0.5
    span = t1 - t0
    outage_time = sum(iv.duration_s for iv in res.outage_intervals)
    assert abs(res.availability - (1.0 - outage_time / span)) < 1e-9
    assert 0.0 < res.availability < 1.0
    assert res.min_margin_db < 0.0


def test_time_varying_rain_enters_budget():
    geom = synthetic_pass()
    rain = [50.0 if 400.0 <= t <= 500.0 else 0.0 for t in geom.t_s]
    cfg = config_with_tca_cn0(geom, 94.0)
    dry = run_pass_budget(geom, cfg)
    wet = run_pass_budget(geom, dataclasses.replace(cfg, rain_rate_mm_h=rain))
    mid = min(range(len(geom.t_s)), key=lambda i: abs(geom.t_s[i] - 450.0))
    assert wet.samples[mid].rain_loss_db > 0.0
    assert wet.samples[0].rain_loss_db == 0.0
    assert wet.samples[mid].margin_db < dry.samples[mid].margin_db
    assert wet.total_volume_mb <= dry.total_volume_mb


def test_modcod_subset_restricts_selection():
    geom = synthetic_pass()
    res = run_pass_budget(geom, config_with_tca_cn0(geom, 94.0, modcods=('BPSK_1_2', 'QPSK_1_2')))
    assert {s.modcod for s in res.samples} <= {'BPSK_1_2', 'QPSK_1_2', None}
    assert max(res.samples, key=lambda s: s.elevation_rad).modcod == 'QPSK_1_2'


def test_volume_and_geometry_validation():
    geom = synthetic_pass()
    res = run_pass_budget(geom, config_with_tca_cn0(geom, 94.0))
    assert res.total_volume_mb > 0.0
    ts = [s.t_s for s in res.samples]
    br = [s.bitrate_bps for s in res.samples]
    expect = sum((b - a) * (x + y) / 2 for a, b, x, y in zip(ts, ts[1:], br, br[1:])) / 8e6
    assert abs(res.total_volume_mb - expect) < 1e-6
    assert res.min_margin_db == min(s.margin_db for s in res.samples)
    try:
        PassGeometry([0.0, 1.0], [1.0], [0.1, 0.2], [0.0, 0.0])
        assert False
    except ValueError:
        pass
    try:
        PassGeometry([0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.0, 0.0])
        assert False
    except ValueError:
        pass
