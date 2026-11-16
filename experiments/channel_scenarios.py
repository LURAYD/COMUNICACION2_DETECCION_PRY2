"""Contrafactuales pareados usando los bloques de señal/canal ya validados."""
import numpy as np
from src.dataset import MAPPERS
from src.waveform import pulse_shape
from src.channels import apply_multipath, apply_carrier_impairments, add_awgn


def scenario_parent(base, seed, index, modulation, snr, scenario):
    # Reconstruir SeedSequence evita consumir streams al cambiar orden de escenarios.
    streams = np.random.SeedSequence([seed, index]).spawn(3)
    params, bits_rng = (np.random.default_rng(s) for s in streams[:2])
    beta = float(params.choice(base['betas']))
    amplitude = float(params.uniform(*base['amplitude_range']))
    phase = float(params.uniform(*base['phase_range_rad']))
    cfo_id = float(params.uniform(*base['id_cfo_range_hz']))
    cfo_shift = float(params.uniform(*base['shift_cfo_abs_range_hz'])) * (-1 if cfo_id < 0 else 1)
    id_index = int(params.integers(len(base['id_channels'])))
    shift_index = int(params.integers(len(base['shift_channels'])))
    timing = int(params.integers(base['sps']))
    bank, channel_index = ('shift_channels', shift_index) if scenario in ('multipath_shift','both_shift') else ('id_channels',id_index)
    cfo = cfo_shift if scenario in ('cfo_shift','both_shift') else (0. if scenario=='cfo_zero' else cfo_id)
    if scenario.startswith('beta_'):
        beta = int(scenario.split('_')[1])/100
    elif scenario not in ('id','cfo_zero','cfo_shift','multipath_shift','both_shift'):
        raise ValueError('Escenario desconocido.')
    pairs = np.array(base[bank][channel_index])
    taps = np.zeros(14,dtype=complex)
    taps[:len(pairs)] = pairs[:,0]+1j*pairs[:,1]
    mapper,k = MAPPERS[modulation]
    bits = bits_rng.integers(0,2,base['symbols_per_parent']*k)
    clean,h = pulse_shape(mapper(bits),base['sps'],beta,base['span'])
    faded = apply_multipath(amplitude*clean,taps)
    signal = apply_carrier_impairments(faded,base['symbol_rate_hz']*base['sps'],phase,cfo)
    received,noise = add_awgn(signal,snr,np.random.default_rng(streams[2]))
    start = len(h)-1+13+timing
    assert start+512 <= (base['symbols_per_parent']-1)*base['sps']
    p = float(np.mean(abs(signal)**2))
    noise_target = p/10**(snr/10)
    window = slice(start,start+512)
    info = dict(parent=f'explore_{seed}_{index}',modulation=modulation,snr_db=snr,scenario=scenario,
                beta=beta,amplitude=amplitude,phase_rad=phase,cfo_hz=cfo,bank=bank,channel_index=channel_index,
                timing=timing,crop_start=start,full_length=len(signal),noise_target=noise_target,
                snr_window_db=float(10*np.log10(np.mean(abs(signal[window])**2)/np.mean(abs(noise[window])**2))))
    return received[window],info,dict(bits=bits,signal=signal[window],noise=noise[window],
                                    standardized_noise=noise[window]/np.sqrt(noise_target))
