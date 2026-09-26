from __future__ import annotations
MODCOD = {'BPSK_1_2': (1, 0.5, 2.0), 'QPSK_1_2': (2, 0.5, 2.5), 'QPSK_3_4': (2, 0.75, 4.5), '8PSK_2_3': (3, 2 / 3, 7.5), '16QAM_3_4': (4, 0.75, 10.5)}

def spectral_efficiency(name):
    bits, rate, _ = MODCOD[name]
    return bits * rate

def required_ebn0(name):
    return MODCOD[name][2]

def choose_modcod(ebn0_db, margin_db=1.0):
    viable = [(spectral_efficiency(n), n) for n in MODCOD if ebn0_db >= required_ebn0(n) + margin_db]
    return max(viable)[1] if viable else None

MODCOD_ORDER = tuple(sorted(MODCOD, key=spectral_efficiency))

def bitrate_bps(name, symbol_rate_sps):
    if symbol_rate_sps <= 0:
        raise ValueError('symbol rate')
    return symbol_rate_sps * spectral_efficiency(name)
