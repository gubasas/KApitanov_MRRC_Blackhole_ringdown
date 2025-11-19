"""Minimal Kapitanov comb detection helper.

This script fetches open LIGO strain data for a given event, extracts a
ringdown window and a background noise window, computes FFT/PSD, and
evaluates a simple frequency-comb detection statistic.

Usage:
  python download_gwpy.py --event GW200129_065458 --detector H1
"""

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import gwosc
import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy import interpolate
import json
import matplotlib.pyplot as plt
import argparse
import math
import os
from tqdm import tqdm


def get_event_gps(event_name: str) -> float:
	if event_name is None:
		raise ValueError('event_name must be provided to get_event_gps()')
	data = event_gps(event_name)
	# `event_gps` may return a float (GPS) or a dict with 'GPS' key.
	if isinstance(data, (float, int)):
		return float(data)
	try:
		return float(data['GPS'])
	except Exception:
		# Fallback: attempt to cast directly
		return float(data)


def fetch_strain(detector: str, start: float, end: float) -> TimeSeries:
	return TimeSeries.fetch_open_data(detector, start, end)


def extract_window(strain: TimeSeries, start_time: float, duration: float) -> TimeSeries:
	# Robust slicing: compute sample indices relative to series start and
	# build a new TimeSeries for the requested interval.
	sampling_rate = strain.sample_rate.value
	# Determine series start time (gps)
	try:
		series_start = float(strain.t0.value)
	except Exception:
		# Fallback to first timestamp value
		series_start = float(strain.times.value[0])
	rel_start = start_time - series_start
	start_idx = int(round(rel_start * sampling_rate))
	n_samples = int(round(duration * sampling_rate))
	data_window = strain.value[start_idx:start_idx + n_samples]
	# Create a TimeSeries for the window; epoch=start_time ensures correct timing
	return TimeSeries(data_window, epoch=start_time, dt=1.0 / sampling_rate)


def compute_psd_from_timeseries(ts, apply_window=True, tukey_alpha=0.25):
	sampling_rate = ts.sample_rate.value
	data = ts.value
	if apply_window:
		window = tukey(len(data), alpha=tukey_alpha)
		data = data * window
	fft_data = fft(data)
	freqs = fftfreq(len(data), d=1.0 / sampling_rate)
	positive = freqs >= 0
	psd = np.abs(fft_data) ** 2
	return freqs[positive], psd[positive]


def compute_fft_psd_fixed_N(data, sampling_rate, N, tukey_alpha=0.25):
	# Prepare data of length N: zero-pad or truncate
	if len(data) < N:
		data_padded = np.pad(data, (0, N - len(data)), mode='constant')
	else:
		data_padded = data[:N]
	window = tukey(N, alpha=tukey_alpha)
	data_windowed = data_padded * window
	fft_data = fft(data_windowed)
	freqs = fftfreq(N, d=1.0 / sampling_rate)
	positive = freqs >= 0
	psd = np.abs(fft_data) ** 2
	return freqs[positive], psd[positive]


def estimate_noise_from_background_segments(bg_ts: TimeSeries, segment_len_samples: int, tukey_alpha=0.25):
	data = bg_ts.value
	n_segments = len(data) // segment_len_samples
	if n_segments < 1:
		raise ValueError('Background too short for requested segment length')
	psd_list = []
	sampling_rate = bg_ts.sample_rate.value
	for i in range(n_segments):
		seg = data[i * segment_len_samples:(i + 1) * segment_len_samples]
		freqs_seg, psd_seg = compute_fft_psd_fixed_N(seg, sampling_rate, segment_len_samples, tukey_alpha=tukey_alpha)
		psd_list.append(psd_seg)
	psd_med = np.median(np.vstack(psd_list), axis=0)
	return freqs_seg, psd_med


def detect_frequency_comb_exact(frequencies, psd, noise_psd, f0, delta_f, n_harmonics=10):
	harmonic_snrs = []
	# By default no masking; wrapper will pass mask params via kwargs when needed
	for n in range(n_harmonics):
		f_expected = f0 + n * delta_f
		bandwidth = delta_f / 4.0
		band_mask = (frequencies >= f_expected - bandwidth) & \
					(frequencies <= f_expected + bandwidth)
		if not np.any(band_mask):
			harmonic_snrs.append(0.0)
			continue
		power_harmonic = np.max(psd[band_mask])
		noise_harmonic = np.median(noise_psd[band_mask])
		snr_harmonic = power_harmonic / noise_harmonic if noise_harmonic > 0 else 0.0
		harmonic_snrs.append(snr_harmonic)

	# Standard detection statistic: sum(SNR_n) / sqrt(n_harmonics)
	detection_statistic = np.sum(harmonic_snrs) / np.sqrt(n_harmonics)
	return detection_statistic, harmonic_snrs


def run_exact_event(event_name: str, detector='H1', ringdown_start=0.01, ringdown_duration=0.09, background_window=(1.0, 5.0), f0=0.0, delta_f=None, n_harmonics=10):
	# Implements the user's exact procedure: 90ms ringdown, 1-5s background, FFT resolution ~11.1 Hz
	gps = get_event_gps(event_name)
	# fetch a bit of data around event
	start = gps - 10.0
	end = gps + 10.0
	strain = fetch_strain(detector, start, end)
	sampling_rate = strain.sample_rate.value

	# Ringdown window: 10-100 ms after merger (90 ms duration)
	ring_start = gps + ringdown_start
	ring_duration = ringdown_duration
	ring_ts = extract_window(strain, ring_start, ring_duration)
	ring_data = ring_ts.value

	# Determine N for target frequency resolution df = 11.1 Hz -> N = fs/df
	target_df = 11.1
	N = int(round(sampling_rate / target_df))
	if N < 1:
		N = len(ring_data)

	freqs_ring, psd_ring = compute_fft_psd_fixed_N(ring_data, sampling_rate, N, tukey_alpha=0.25)

	# Background: 1-5 seconds before merger
	bg_start = gps - background_window[1]
	bg_end = gps - background_window[0]
	bg_ts = fetch_strain(detector, bg_start, bg_end)
	# Use same segment length (N samples) across background to build noise PSD
	segment_len_samples = N
	freqs_bg, noise_psd = estimate_noise_from_background_segments(bg_ts, segment_len_samples, tukey_alpha=0.25)

	# Use exact detection procedure
	if delta_f is None:
		delta_f_use = 1.0
	else:
		delta_f_use = delta_f
	detection_stat, harmonic_snrs = detect_frequency_comb_exact(freqs_ring, psd_ring, noise_psd, f0, delta_f_use, n_harmonics=n_harmonics)

	return {
		'event': event_name,
		'gps': gps,
		'freqs': freqs_ring,
		'psd': psd_ring,
		'freqs_bg': freqs_bg,
		'noise_psd': noise_psd,
		'detection_stat': float(detection_stat),
		'harmonic_snrs': list(map(float, harmonic_snrs)),
		'sampling_rate': sampling_rate,
	}


def compute_welch_psd_from_timeseries(ts, nperseg=None, noverlap=None, window='hann'):
	sampling_rate = ts.sample_rate.value
	data = ts.value
	if nperseg is None:
		# choose nperseg ~ length/2 or up to 4096
		nperseg = min(len(data) // 2, 4096)
	freqs, psd = welch(data, fs=sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)
	return freqs, psd


def estimate_noise_background(noise_ts: TimeSeries, target_len: int, tukey_alpha=0.25):
	# Use Welch PSD across the background interval for robust noise estimate.
	freqs, psd = compute_welch_psd_from_timeseries(noise_ts, nperseg=None)
	return freqs, psd


def detect_frequency_comb(frequencies, psd, noise_psd, f0, delta_f, n_harmonics=10, noise_freqs=None, mask_lines=False, mask_width=5.0, known_lines=None):
	# detect comb with optional masking of known instrumental line frequencies
	if known_lines is None:
		# default: use 60 Hz power-line harmonics up to Nyquist
		maxf = float(np.max(frequencies)) if len(frequencies) > 0 else 2000.0
		known_lines = np.arange(60.0, maxf + 60.0, 60.0)

	harmonic_snrs = []
	used_mask_flags = []
	for n in range(n_harmonics):
		f_expected = f0 + n * delta_f
		# If masking enabled, skip harmonics that fall within mask_width of any known line
		if mask_lines:
			near = np.any(np.abs(np.array(known_lines) - f_expected) <= mask_width)
			if near:
				# mark as masked and append zero SNR
				harmonic_snrs.append(0.0)
				used_mask_flags.append(True)
				continue
			used_mask_flags.append(False)

		bandwidth = delta_f / 4.0
		band_mask = (frequencies >= f_expected - bandwidth) & (frequencies <= f_expected + bandwidth)
		if not np.any(band_mask):
			harmonic_snrs.append(0.0)
			continue
		power_harmonic = np.max(psd[band_mask])
		# If noise_psd is on a different frequency grid, interpolate it onto `frequencies`.
		if noise_freqs is not None and len(noise_freqs) != len(frequencies):
			interp_fn = interpolate.interp1d(noise_freqs, noise_psd, bounds_error=False, fill_value=np.median(noise_psd))
			noise_on_target = interp_fn(frequencies)
			noise_harmonic = np.median(noise_on_target[band_mask])
		else:
			# assume same grid
			noise_harmonic = np.median(noise_psd[band_mask])
		if noise_harmonic <= 0:
			snr_harmonic = 0.0
		else:
			snr_harmonic = power_harmonic / noise_harmonic
		harmonic_snrs.append(float(snr_harmonic))

	harmonic_snrs = np.array(harmonic_snrs)
	# Adjust denominator to number of unmasked harmonics if masking used
	if mask_lines:
		n_used = float(np.sum(~np.array(used_mask_flags))) if len(used_mask_flags) > 0 else float(n_harmonics)
		if n_used <= 0:
			return 0.0, harmonic_snrs.tolist()
		detection_statistic = float(np.sum(harmonic_snrs) / math.sqrt(n_used))
	else:
		detection_statistic = float(np.sum(harmonic_snrs) / math.sqrt(n_harmonics))

	return detection_statistic, harmonic_snrs.tolist()


def bootstrap_background_pvalue(background_ts: TimeSeries, ring_len_samples: int, detector, event_detector_strain_start, event_detector_strain_end, f0, delta_f, n_harmonics=10, n_boot=500):
	# Build distribution of detection_statistic from many background segments
	sampling_rate = background_ts.sample_rate.value
	data = background_ts.value
	max_start = len(data) - ring_len_samples
	if max_start <= 0:
		return None, None
	stats = []
	rng = np.random.default_rng(12345)
	for _ in range(n_boot):
		s = int(rng.integers(0, max_start + 1))
		seg = data[s:s + ring_len_samples]
		ts_seg = TimeSeries(seg, epoch=0.0, dt=1.0 / sampling_rate)
		freqs_seg, psd_seg = compute_welch_psd_from_timeseries(ts_seg, nperseg=None)
		# Resample noise PSD to same freqs (use freqs_seg for both)
		detection_stat, _ = detect_frequency_comb(freqs_seg, psd_seg, psd_seg, f0, delta_f, n_harmonics=n_harmonics)
		stats.append(detection_stat)
	return np.array(stats), sampling_rate


def resample_psd_to_common_axis(freqs, psd, target_freqs):
	interp = interpolate.interp1d(freqs, psd, bounds_error=False, fill_value=0.0)
	return interp(target_freqs)


def stack_events_on_mass_normalized_axis(event_results, reference_mass=55.5, df=1.0, max_freq=2000.0):
	target_freqs = np.arange(0.0, max_freq, df)
	stacked = np.zeros_like(target_freqs)
	weights = np.zeros_like(target_freqs)
	for ev in event_results:
		mass = ev.get('mass', reference_mass)
		scale = mass / reference_mass
		freqs_scaled = ev['freqs'] * scale
		psd_scaled = resample_psd_to_common_axis(freqs_scaled, ev['psd'], target_freqs)
		weight = 1.0
		stacked += psd_scaled * weight
		weights += (psd_scaled > 0) * weight
	weights[weights == 0] = 1.0
	stacked_norm = stacked / weights
	return target_freqs, stacked_norm


def run_single_event(event_name: str, detector='H1', ringdown_start=0.01, ringdown_duration=0.09, background_window=(1.0, 5.0), f0=0.0, delta_f=None, n_harmonics=10):
	gps = get_event_gps(event_name)
	start = gps - 10.0
	end = gps + 10.0
	strain = fetch_strain(detector, start, end)
	sampling_rate = strain.sample_rate.value
	ring_start = gps + ringdown_start
	ring_ts = extract_window(strain, ring_start, ringdown_duration)
	# Compute both FFT-based and Welch PSDs; detection will use Welch PSD
	freqs_fft, psd_fft = compute_psd_from_timeseries(ring_ts)
	freqs, psd = compute_welch_psd_from_timeseries(ring_ts)

	bg_start = gps - background_window[1]
	bg_end = gps - background_window[0]
	bg_ts = fetch_strain(detector, bg_start, bg_end)
	freqs_bg, noise_psd = estimate_noise_background(bg_ts, target_len=len(ring_ts))

	detection_stat, harmonic_snrs = detect_frequency_comb(freqs, psd, noise_psd, f0, delta_f if delta_f else 1.0, n_harmonics=n_harmonics, noise_freqs=freqs_bg)

	# Bootstrap p-value estimate from background segments (fast approximation)
	ring_len_samples = len(ring_ts)
	bg_stats, _ = bootstrap_background_pvalue(bg_ts, ring_len_samples, detector, bg_start, bg_end, f0, delta_f if delta_f else 1.0, n_harmonics=n_harmonics, n_boot=400)
	pvalue = None
	if bg_stats is not None:
		pvalue = float(np.mean(bg_stats >= detection_stat))
	result = {
		'event': event_name,
		'gps': gps,
		'freqs': freqs,
		'psd': psd,
		'freqs_bg': freqs_bg,
		'noise_psd': noise_psd,
		'detection_stat': detection_stat,
		'harmonic_snrs': harmonic_snrs,
		'sampling_rate': sampling_rate,
	}
	if pvalue is not None:
		result['pvalue'] = pvalue
		result['bg_stats'] = bg_stats.tolist()
	return result


def compute_delta_f_from_mass(mass_solar: float, k: int) -> float:
	# Δf = c^3 / (4*k * π * G * M)
	# mass_solar: mass in solar masses
	G = 6.67430e-11
	c = 299792458.0
	M_sun = 1.98847e30
	M = mass_solar * M_sun
	denom = 4.0 * k * math.pi * G * M
	return (c ** 3) / denom


def run_validation(event_name: str, detectors=('H1', 'L1'), ringdown_starts=(0.01, 0.02, 0.03), ringdown_durations=(0.05, 0.09, 0.12), k_values=(2, 3, 4), pre_merger_offsets=(-4.0, -3.0, -2.0, -1.0), n_harmonics=10, mass_override=None):
	"""Run validation tests A. L1 confirmation, time-window robustness, k-scan, and pre-merger checks.

	Saves a JSON summary to `results/validation_{event}.json`.
	"""
	# Default mass map from user's priority list (solar masses)
	default_mass_map = {
		'GW200129_065458': 60.2,
		'GW200224_222234': 68.7,
		'GW200112_155838': 60.8,
		'GW191216_213338': 18.9,
		'GW200311_115853': 59.0,
	}
	mass = mass_override if mass_override is not None else default_mass_map.get(event_name)

	summary = {'event': event_name, 'validation': {}}

	# A1: Detector confirmation (H1/L1)
	det_results = {}
	for det in detectors:
		try:
			res = run_exact_event(event_name, detector=det, ringdown_start=0.01, ringdown_duration=0.09, delta_f=None, n_harmonics=n_harmonics)
			det_results[det] = {'detection_stat': res.get('detection_stat'), 'harmonic_snrs': res.get('harmonic_snrs')}
		except Exception as e:
			det_results[det] = {'error': str(e)}
	summary['validation']['detector_confirmation'] = det_results

	# A2: Time window robustness
	time_grid_results = []
	for start in ringdown_starts:
		for dur in ringdown_durations:
			try:
				res = run_exact_event(event_name, detector=detectors[0], ringdown_start=start, ringdown_duration=dur, delta_f=None, n_harmonics=n_harmonics)
				time_grid_results.append({'start': start, 'duration': dur, 'detection_stat': res.get('detection_stat'), 'harmonic_snrs': res.get('harmonic_snrs')})
			except Exception as e:
				time_grid_results.append({'start': start, 'duration': dur, 'error': str(e)})
	summary['validation']['time_window_robustness'] = time_grid_results

	# A3: k-value comparison (compute Δf from mass for k in k_values)
	k_results = {}
	if mass is not None:
		for k in k_values:
			delta_f_k = compute_delta_f_from_mass(mass, k)
			try:
				res = run_exact_event(event_name, detector=detectors[0], ringdown_start=0.01, ringdown_duration=0.09, delta_f=delta_f_k, n_harmonics=n_harmonics)
				k_results[f'k={k}'] = {'delta_f': delta_f_k, 'detection_stat': res.get('detection_stat'), 'harmonic_snrs': res.get('harmonic_snrs')}
			except Exception as e:
				k_results[f'k={k}'] = {'delta_f': delta_f_k, 'error': str(e)}
	else:
		k_results['error'] = 'mass unknown; provide mass_override to compute Δf from mass'
	summary['validation']['k_value_comparison'] = k_results

	# A4: Background noise analysis — compute detection statistic on pre-merger offsets
	pre_results = []
	for offset in pre_merger_offsets:
		try:
			res = run_exact_event(event_name, detector=detectors[0], ringdown_start=offset, ringdown_duration=0.09, delta_f=None, n_harmonics=n_harmonics)
			pre_results.append({'offset': offset, 'detection_stat': res.get('detection_stat'), 'harmonic_snrs': res.get('harmonic_snrs')})
		except Exception as e:
			pre_results.append({'offset': offset, 'error': str(e)})
	summary['validation']['pre_merger_background'] = pre_results

	# Save summary
	os.makedirs('results', exist_ok=True)
	outpath = os.path.join('results', f'validation_{event_name}.json')
	with open(outpath, 'w') as fh:
		json.dump(summary, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

	return summary


def load_event_for_bootstrap(event_name: str, detector='H1', ringdown_start=0.01, ringdown_duration=0.09, background_window=(1.0, 5.0)):
	gps = get_event_gps(event_name)
	start = gps - 10.0
	end = gps + 10.0
	strain = fetch_strain(detector, start, end)
	sampling_rate = strain.sample_rate.value
	ring_start = gps + ringdown_start
	ring_ts = extract_window(strain, ring_start, ringdown_duration)
	ring_data = ring_ts.value

	bg_start = gps - background_window[1]
	bg_end = gps - background_window[0]
	bg_ts = fetch_strain(detector, bg_start, bg_end)
	return {
		'gps': gps,
		'strain': strain,
		'ring_ts': ring_ts,
		'ring_data': ring_data,
		'bg_ts': bg_ts,
		'sampling_rate': sampling_rate,
	}


def bootstrap_exact_from_background(bg_ts: TimeSeries, sampling_rate: float, segment_len_samples: int, noise_psd_med, f0, delta_f, n_harmonics=10, n_boot=500):
	data = bg_ts.value
	max_start = len(data) - segment_len_samples
	if max_start <= 0:
		return None
	rng = np.random.default_rng(123456)
	stats = []
	for _ in range(n_boot):
		s = int(rng.integers(0, max_start + 1))
		seg = data[s:s + segment_len_samples]
		freqs_seg, psd_seg = compute_fft_psd_fixed_N(seg, sampling_rate, segment_len_samples, tukey_alpha=0.25)
		# use masked detection if caller provided a masked noise_psd (caller controls masking by passing correct delta_f and mask options)
		stat, _ = detect_frequency_comb_exact(freqs_seg, psd_seg, noise_psd_med, f0, delta_f, n_harmonics=n_harmonics)
		stats.append(stat)
	return np.array(stats)


def run_bootstrap_for_event(event_name: str, detector='H1', k=2, n_boot=500, n_harmonics=10):
	# compute delta_f from mass map
	default_mass_map = {
		'GW200129_065458': 60.2,
		'GW200224_222234': 68.7,
		'GW200112_155838': 60.8,
		'GW191216_213338': 18.9,
		'GW200311_115853': 59.0,
	}
	mass = default_mass_map.get(event_name)
	if mass is None:
		raise ValueError('Mass unknown for event; provide mass_override')
	delta_f = compute_delta_f_from_mass(mass, k)

	data = load_event_for_bootstrap(event_name, detector=detector)
	sampling_rate = data['sampling_rate']
	ring_data = data['ring_data']
	bg_ts = data['bg_ts']

	# N for df~11.1 Hz
	N = int(round(sampling_rate / 11.1))
	freqs_ring, psd_ring = compute_fft_psd_fixed_N(ring_data, sampling_rate, N, tukey_alpha=0.25)
	freqs_bg, noise_psd_med = estimate_noise_from_background_segments(bg_ts, N, tukey_alpha=0.25)

	real_stat, harmonic_snrs = detect_frequency_comb_exact(freqs_ring, psd_ring, noise_psd_med, 0.0, delta_f, n_harmonics=n_harmonics)

	stats_bg = bootstrap_exact_from_background(bg_ts, sampling_rate, N, noise_psd_med, 0.0, delta_f, n_harmonics=n_harmonics, n_boot=n_boot)
	pvalue = None
	if stats_bg is not None:
		pvalue = float(np.mean(stats_bg >= real_stat))

	out = {
		'event': event_name,
		'detector': detector,
		'k': k,
		'delta_f': delta_f,
		'real_stat': float(real_stat),
		'harmonic_snrs': harmonic_snrs,
		'bootstrap_stats': stats_bg.tolist() if (stats_bg is not None and hasattr(stats_bg, 'tolist')) else (list(stats_bg) if stats_bg is not None else None),
		'pvalue': pvalue,
	}
	os.makedirs('results', exist_ok=True)
	outpath = os.path.join('results', f'{event_name}_{detector}_bootstrap_k{k}.json')
	with open(outpath, 'w') as fh:
		json.dump(out, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
	return out


def run_stacking(events, detectors_map=None, k=2, df=1.0, max_freq=2000.0, n_boot=500, n_harmonics=10):
	# detectors_map: dict event->detector choice
	default_mass_map = {
		'GW200129_065458': 60.2,
		'GW200224_222234': 68.7,
		'GW200112_155838': 60.8,
		'GW191216_213338': 18.9,
		'GW200311_115853': 59.0,
	}
	if detectors_map is None:
		detectors_map = {ev: 'H1' for ev in events}

	# Load ring PSDs and masses. Keep per-event N and sampling_rate for bootstrapping.
	ev_results = []
	for ev in events:
		det = detectors_map.get(ev, 'H1')
		# Prefer loading saved exact-mode result JSON if available to avoid re-downloading
		saved = None
		for try_det in (det, 'H1', 'L1'):
			path = os.path.join('results', f"{ev}_{try_det}_result.json")
			if os.path.exists(path):
				try:
					with open(path, 'r') as fh:
						saved = json.load(fh)
					det = try_det
					break
				except Exception:
					saved = None
		if saved is not None:
			freqs_ring = np.array(saved['freqs'])
			psd_ring = np.array(saved['psd'])
			sampling_rate = saved.get('sampling_rate', None)
			N = int(round(sampling_rate / 11.1)) if sampling_rate is not None else len(freqs_ring) * 2
			mass = default_mass_map.get(ev)
			ev_results.append({'event': ev, 'mass': mass, 'freqs': freqs_ring, 'psd': psd_ring, 'bg_ts': None, 'detector': det, 'N': N, 'sampling_rate': sampling_rate})
			continue

		# Fallback: fetch data live
		try:
			data = load_event_for_bootstrap(ev, detector=det)
		except Exception:
			# If we can't load live, skip this event
			continue
		sampling_rate = data['sampling_rate']
		N = int(round(sampling_rate / 11.1))
		freqs_ring, psd_ring = compute_fft_psd_fixed_N(data['ring_data'], sampling_rate, N, tukey_alpha=0.25)
		mass = default_mass_map.get(ev)
		ev_results.append({'event': ev, 'mass': mass, 'freqs': freqs_ring, 'psd': psd_ring, 'bg_ts': data['bg_ts'], 'detector': det, 'N': N, 'sampling_rate': sampling_rate})

	if len(ev_results) == 0:
		raise RuntimeError('No event results available for stacking')

	# Stack (mass-normalized)
	target_freqs, stacked = stack_events_on_mass_normalized_axis(ev_results, reference_mass=55.5, df=df, max_freq=max_freq)

	# Compute detection on stacked PSD
	ref_mass = 55.5
	delta_f = compute_delta_f_from_mass(ref_mass, k)
	median_noise = float(np.median(stacked)) if np.any(stacked) else 1.0
	noise_array = np.full_like(target_freqs, median_noise, dtype=float)
	stacked_stat, stacked_harmonics = detect_frequency_comb_exact(target_freqs, stacked, noise_array, 0.0, delta_f, n_harmonics=n_harmonics)

	# Bootstrap stacks: for each bootstrap iteration, pick random bg segment for each event, mass-normalize and stack
	rng = np.random.default_rng(12345)
	boot_stats = []
	for i in range(n_boot):
		stack_i = np.zeros_like(target_freqs, dtype=float)
		for evr in ev_results:
			bg_ts = evr.get('bg_ts')
			# If no background TimeSeries available, try to fetch it
			if bg_ts is None:
				try:
					_ld = load_event_for_bootstrap(evr['event'], detector=evr.get('detector', 'H1'))
					bg_ts = _ld.get('bg_ts')
				except Exception:
					bg_ts = None

			N_ev = evr.get('N', None)
			sampling_rate = evr.get('sampling_rate', None)
			if N_ev is None or sampling_rate is None:
				# Fallback: derive N from length of stored freqs (approx)
				N_ev = len(evr['freqs']) * 2
				sampling_rate = evr.get('sampling_rate', 4096.0)

			if bg_ts is None:
				# no background available; use zeros for this event
				seg = np.zeros(N_ev)
			else:
				bg = bg_ts.value
				max_start = len(bg) - N_ev
				if max_start <= 0:
					seg = np.zeros(N_ev)
				else:
					s = int(rng.integers(0, max_start + 1))
					seg = bg[s:s + N_ev]

			freqs_seg, psd_seg = compute_fft_psd_fixed_N(seg, sampling_rate, N_ev, tukey_alpha=0.25)
			mass = evr['mass'] if evr.get('mass') is not None else ref_mass
			scale = mass / ref_mass
			freqs_scaled = freqs_seg * scale
			psd_scaled = resample_psd_to_common_axis(freqs_scaled, psd_seg, target_freqs)
			stack_i += psd_scaled

		stat_i, _ = detect_frequency_comb_exact(target_freqs, stack_i, np.full_like(target_freqs, np.median(stack_i) if np.any(stack_i) else median_noise), 0.0, delta_f, n_harmonics=n_harmonics)
		boot_stats.append(float(stat_i))

	pvalue = float(np.mean(np.array(boot_stats) >= stacked_stat))
	out = {'events': events, 'k': k, 'delta_f_ref': delta_f, 'stacked_stat': float(stacked_stat), 'stacked_harmonics': stacked_harmonics, 'bootstrap_stats': boot_stats, 'pvalue': pvalue}
	os.makedirs('results', exist_ok=True)
	with open(os.path.join('results', f'stacking_k{k}.json'), 'w') as fh:
		json.dump(out, fh, indent=2)
	return out


def cli():
	parser = argparse.ArgumentParser(description='Minimal Kapitanov comb detection pipeline')
	parser.add_argument('--event', required=False, help='GW event name as in GWTC (e.g. GW200129_065458)')
	parser.add_argument('--detector', default='H1')
	parser.add_argument('--ringdown-start', type=float, default=0.01, help='Seconds after merger to start ringdown window')
	parser.add_argument('--ringdown-duration', type=float, default=0.09, help='Ringdown window duration (seconds)')
	parser.add_argument('--delta-f', type=float, default=None, help='Comb spacing in Hz (if known). If not set, provide mass-based calculation externally.')
	parser.add_argument('--f0', type=float, default=0.0, help='Base frequency f0 (Hz)')
	parser.add_argument('--n-harmonics', type=int, default=8)
	parser.add_argument('--exact', action='store_true', help='Run exact-mode pipeline matching provided procedure')
	parser.add_argument('--validate', action='store_true', help='Run validation suite (detector confirmation, time robustness, k-scan, pre-merger)')
	parser.add_argument('--detectors', type=str, default='H1,L1', help='Comma-separated detectors to test')
	parser.add_argument('--ringdown-starts', type=str, default='0.01,0.02,0.03', help='Comma-separated ringdown starts (s)')
	parser.add_argument('--ringdown-durations', type=str, default='0.05,0.09,0.12', help='Comma-separated ringdown durations (s)')
	parser.add_argument('--k-values', type=str, default='2,3,4', help='Comma-separated k values to test')
	parser.add_argument('--pre-merger-offsets', type=str, default='-4.0,-3.0,-2.0,-1.0', help='Comma-separated pre-merger offsets (s)')
	parser.add_argument('--mass-override', type=float, default=None, help='Override mass (solar masses) for Δf computation')
	parser.add_argument('--do-bootstrap', action='store_true', help='Run per-event bootstrap for a list of events (use with --events)')
	parser.add_argument('--do-stack', action='store_true', help='Run stacking + bootstrap for a list of events (use with --events)')
	parser.add_argument('--kvalue', type=int, default=2, help='k value to use for delta_f computation when running bootstrap/stack')
	parser.add_argument('--events', type=str, default=None, help='Comma-separated event names for bootstrap/stacking')
	parser.add_argument('--bootstrap-iters', type=int, default=500, help='Number of bootstrap iterations per event')
	parser.add_argument('--stack-iters', type=int, default=500, help='Number of bootstrap iterations for stacking')
	parser.add_argument('--mask-lines', action='store_true', help='Mask known instrumental line frequencies (±mask-width Hz) when computing detection')
	parser.add_argument('--mask-width', type=float, default=5.0, help='Half-width in Hz for masking around known lines')
	args = parser.parse_args()

	if args.delta_f is None:
		print('Warning: --delta-f not provided. Detection will use delta_f=1.0 Hz as placeholder.')
	# If orchestration requested, run bootstrap/stacking without requiring --event
	if args.do_bootstrap or args.do_stack:
		if not args.events:
			print('Error: --events must be provided as comma-separated list when using --do-bootstrap or --do-stack')
			return
		event_list = [e.strip() for e in args.events.split(',') if e.strip()]
		if args.do_bootstrap:
			print('Running bootstrap for events:', event_list)
			for ev in event_list:
				for det in ('H1', 'L1'):
					try:
						out = run_bootstrap_for_event(ev, detector=det, k=args.kvalue, n_boot=args.bootstrap_iters, n_harmonics=args.n_harmonics)
						print(f"Saved bootstrap for {ev} {det}: p={out.get('pvalue')}")
					except Exception as e:
						print(f"Bootstrap failed for {ev} {det}: {e}")
		if args.do_stack:
			print('Running stacking for events:', event_list)
			det_map = {ev: 'H1' for ev in event_list}
			stack_out = run_stacking(event_list, detectors_map=det_map, k=args.kvalue, n_boot=args.stack_iters, n_harmonics=args.n_harmonics)
			print('Stacking saved, p-value:', stack_out.get('pvalue'))
		return

	# For single-event operations (validate/exact/single) require --event
	if not args.event:
		# If results folder contains saved example results, load the first one and print a short summary
		results_dir = 'results'
		if os.path.isdir(results_dir):
			# look for any *_result.json files
			files = sorted([f for f in os.listdir(results_dir) if f.endswith('_result.json')])
			if len(files) > 0:
				sample = files[0]
				print(f"No --event provided. Showing sample result from {os.path.join(results_dir, sample)}")
				try:
					with open(os.path.join(results_dir, sample), 'r') as fh:
						res = json.load(fh)
					# print the summary
					print('Event:', res.get('event'))
					print('GPS:', res.get('gps'))
					print('Detection statistic:', res.get('detection_stat'))
					print('Harmonic SNRs:', res.get('harmonic_snrs'))
				except Exception as e:
					print('Failed to load sample result:', e)
				return
		# no sample results available; print helpful usage and exit
		print('Error: --event must be provided for single-event operations. Use --do-bootstrap/--do-stack with --events for batch runs, or place a result JSON in results/ to view a sample when running with no args.')
		return

	print(f"Fetching and analyzing event {args.event} on detector {args.detector}")
	if args.validate:
		# parse lists
		det_list = [d.strip() for d in args.detectors.split(',') if d.strip()]
		starts = [float(x) for x in args.ringdown_starts.split(',') if x.strip()]
		durations = [float(x) for x in args.ringdown_durations.split(',') if x.strip()]
		kvals = [int(x) for x in args.k_values.split(',') if x.strip()]
		pre_offsets = [float(x) for x in args.pre_merger_offsets.split(',') if x.strip()]
		print(f'Running validation for {args.event} detectors={det_list} starts={starts} durations={durations} k={kvals} pre_offsets={pre_offsets}')
		summary = run_validation(args.event, detectors=tuple(det_list), ringdown_starts=tuple(starts), ringdown_durations=tuple(durations), k_values=tuple(kvals), pre_merger_offsets=tuple(pre_offsets), n_harmonics=args.n_harmonics, mass_override=args.mass_override)
		print('Validation saved:', os.path.join('results', f'validation_{args.event}.json'))
		return
	elif args.exact:
		res = run_exact_event(args.event, detector=args.detector, ringdown_start=args.ringdown_start, ringdown_duration=args.ringdown_duration, delta_f=args.delta_f, f0=args.f0, n_harmonics=args.n_harmonics)
	else:
		res = run_single_event(args.event, detector=args.detector, ringdown_start=args.ringdown_start, ringdown_duration=args.ringdown_duration, delta_f=args.delta_f, f0=args.f0, n_harmonics=args.n_harmonics)
	print('Detection statistic:', res.get('detection_stat'))
	print('Harmonic SNRs:', res.get('harmonic_snrs'))
	# Save result
	os.makedirs('results', exist_ok=True)
	outpath = os.path.join('results', f"{args.event}_{args.detector}_result.json")
	with open(outpath, 'w') as fh:
		def _json_default(o):
			try:
				# numpy arrays
				if hasattr(o, 'tolist'):
					return o.tolist()
				if isinstance(o, (np.floating, np.integer)):
					return o.item()
			except Exception:
				pass
			raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

		json.dump(res, fh, indent=2, default=_json_default)
	print(f"Saved result to {outpath}")

	# Orchestration: bootstrap / stacking for multiple events
	if args.do_bootstrap or args.do_stack:
		if not args.events:
			print('Error: --events must be provided as comma-separated list when using --do-bootstrap or --do-stack')
		else:
			event_list = [e.strip() for e in args.events.split(',') if e.strip()]
			if args.do_bootstrap:
				print('Running bootstrap for events:', event_list)
				for ev in event_list:
					for det in ('H1', 'L1'):
						try:
							out = run_bootstrap_for_event(ev, detector=det, k=args.kvalue, n_boot=args.bootstrap_iters, n_harmonics=args.n_harmonics)
							print(f"Saved bootstrap for {ev} {det}: p={out.get('pvalue')}")
						except Exception as e:
							print(f"Bootstrap failed for {ev} {det}: {e}")
			if args.do_stack:
				print('Running stacking for events:', event_list)
				# choose detectors: prefer H1 if available, else L1 — simple mapping
				det_map = {ev: 'H1' for ev in event_list}
				stack_out = run_stacking(event_list, detectors_map=det_map, k=args.kvalue, n_boot=args.stack_iters, n_harmonics=args.n_harmonics)
				print('Stacking saved, p-value:', stack_out.get('pvalue'))


if __name__ == '__main__':
	cli()
