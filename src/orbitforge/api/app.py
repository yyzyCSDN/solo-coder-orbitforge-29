from __future__ import annotations
import math
from dataclasses import asdict
from fastapi import FastAPI
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.link.modulation import SwitchPolicy
from orbitforge.link.pass_budget import LinkConfig, PassSample, evaluate_pass
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.storage.sqlite import Store
app = FastAPI(title='OrbitForge Mission Lab', version='1.0.0')
store = Store(':memory:')

class KeplerReq(BaseModel):
    mean_anomaly: float
    eccentricity: float

class HohmannReq(BaseModel):
    r1_km: float
    r2_km: float

class LinkReq(BaseModel):
    range_km: float
    freq_hz: float

class PassSampleReq(BaseModel):
    t: float
    range_km: float
    elevation_rad: float
    range_rate_km_s: float = 0.0

class LinkPassReq(BaseModel):
    samples: list[PassSampleReq]
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
    required_margin_db: float = 1.0
    hysteresis_db: float = 1.0
    min_dwell_s: float = 30.0

class EclipseReq(BaseModel):
    sat: list[float]
    sun: list[float]

class RotateReq(BaseModel):
    q: list[float]
    v: list[float]

@app.get('/live')
def live():
    return {'status': 'live'}

@app.get('/ready')
def ready():
    return {'status': 'ready'}

@app.post('/v1/orbit/kepler')
def kepler(r: KeplerReq):
    return {'eccentric_anomaly': solve_kepler_elliptic(r.mean_anomaly, r.eccentricity)}

@app.post('/v1/maneuver/hohmann')
def h(r: HohmannReq):
    return hohmann(r.r1_km, r.r2_km)

@app.post('/v1/link/fspl')
def l(r: LinkReq):
    return {'loss_db': free_space_loss_db(r.range_km, r.freq_hz)}

def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

@app.post('/v1/link/pass')
def link_pass(r: LinkPassReq):
    config = LinkConfig(freq_hz=r.freq_hz, tx_power_dbw=r.tx_power_dbw, tx_gain_dbi=r.tx_gain_dbi,
                        rx_gain_dbi=r.rx_gain_dbi, system_temp_k=r.system_temp_k,
                        bandwidth_hz=r.bandwidth_hz, fixed_losses_db=r.fixed_losses_db,
                        humidity=r.humidity, rain_rate_mm_h=r.rain_rate_mm_h,
                        receiver_max_doppler_hz=r.receiver_max_doppler_hz,
                        policy=SwitchPolicy(r.required_margin_db, r.hysteresis_db, r.min_dwell_s))
    result = evaluate_pass([PassSample(s.t, s.range_km, s.elevation_rad, s.range_rate_km_s)
                            for s in r.samples], config)
    return _jsonable({
        'points': [asdict(p) for p in result.points],
        'switches': [asdict(e) for e in result.switches],
        'outage_windows': [asdict(w) for w in result.outage_windows],
        'low_margin_windows': [asdict(w) for w in result.low_margin_windows],
        'total_volume_mb': result.total_volume_mb,
    })

@app.post('/v1/environment/eclipse')
def e(r: EclipseReq):
    return {'state': eclipse_state(Vec3(*r.sat), Vec3(*r.sun))}

@app.post('/v1/attitude/rotate')
def rotate(r: RotateReq):
    q = Quaternion(*r.q)
    v = q.rotate(Vec3(*r.v))
    return {'v': v.as_tuple()}

@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
