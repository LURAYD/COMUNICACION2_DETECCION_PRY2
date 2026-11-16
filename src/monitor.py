"""Monitor integrado de cinco salidas y procesador CLI de señales desconocidas (Fase 7).

Proporciona las cinco salidas obligatorias del proyecto:
1. Detección de presencia (Sí / No) mediante estadística de energía y umbral calibrado.
2. Densidad espectral de potencia (PSD) y estimación de ancho de banda ocupado al 99%.
3. Estimación de SNR condicionada (o estado formal si no hay referencia de piso de ruido).
4. Tipo de modulación reconocida (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 4PAM).
5. Confianza calibrada de la predicción (probabilidad empírica, no softmax crudo).
"""
import argparse
import json
from pathlib import Path
from numbers import Real
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.detection import detect_energy, energy_statistic, calibrate_energy
from src.spectrum import estimate_spectrum
from src.snr_estimation import estimate_snr, calibrate_noise_reference


# Constantes del proyecto
CLASSES = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM")
DEFAULT_SVM_MODEL = Path("results/fase5/conventional_256.joblib")
DEFAULT_POLICY = Path("results/adaptive/policy.joblib")
DEFAULT_ENERGY_THRESHOLD = 1.15239  # Umbral Fase 4 E1 para Pfa=0.01


def load_signal(file_path, dtype="complex64"):
    """Carga una observación desde formatos comunes (.c64, .bin, .npy, .npz, .csv).

    Devuelve un array 1D de números complejos finitos.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Archivo de señal no encontrado: {path}")

    ext = path.suffix.lower()

    if ext in (".c64", ".bin"):
        dt = np.complex64 if dtype == "complex64" else np.dtype(dtype)
        samples = np.fromfile(str(path), dtype=dt)

    elif ext == ".npy":
        arr = np.load(str(path))
        if np.iscomplexobj(arr):
            samples = arr.ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            samples = arr[:, 0] + 1j * arr[:, 1]
        elif arr.ndim == 2 and arr.shape[0] == 2:
            samples = arr[0, :] + 1j * arr[1, :]
        else:
            samples = arr.ravel().astype(np.complex64)

    elif ext == ".npz":
        with np.load(str(path)) as data:
            target_key = None
            for key in ("iq", "iq256", "iq512", "samples", "data", "x", "y"):
                if key in data:
                    target_key = key
                    break
            if target_key is None:
                target_key = data.files[0]
            arr = data[target_key]
            if np.iscomplexobj(arr):
                samples = arr.ravel()
            elif arr.ndim == 2 and arr.shape[1] == 2:
                samples = arr[:, 0] + 1j * arr[:, 1]
            else:
                samples = arr.ravel().astype(np.complex64)

    elif ext == ".csv":
        # Puede tener columnas I, Q o una columna de texto con números complejos
        try:
            arr = np.genfromtxt(str(path), delimiter=",", comments="#")
            if arr.ndim == 2 and arr.shape[1] >= 2:
                samples = arr[:, 0] + 1j * arr[:, 1]
            else:
                samples = arr.ravel().astype(np.complex64)
        except Exception:
            # Intentar lectura como cadenas complejas
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            samples = np.array([complex(x.replace(" ", "")) for x in lines], dtype=np.complex64)

    else:
        # Fallback a binario complex64
        samples = np.fromfile(str(path), dtype=np.complex64)

    if samples.size < 2:
        raise ValueError(f"La señal en {path} tiene menos de dos muestras.")
    if not np.isfinite(samples).all():
        raise ValueError(f"La señal en {path} contiene valores infinitos o NaN.")

    return np.asarray(samples, dtype=np.complex128)


def analyze(iq, fs_hz=None, detection_calibration=None, noise_reference=None, nfft=None):
    """Integración original de Fase 4 (mantiene retrocompatibilidad)."""
    detection = (
        detect_energy(iq, detection_calibration)
        if detection_calibration is not None
        else {"detected": None, "energy": None, "threshold": None}
    )
    return {
        "detection": detection,
        "spectrum": estimate_spectrum(iq, fs_hz=fs_hz, nfft=nfft),
        "snr": estimate_snr(iq, reference=noise_reference, detected=detection["detected"]),
    }


def analyze_complete(
    iq,
    signal_id="unknown_signal",
    fs_hz=8000.0,
    model_path=None,
    policy_path=None,
    detection_calibration=None,
    noise_reference=None,
    default_threshold=DEFAULT_ENERGY_THRESHOLD,
    plot_path=None,
):
    """Ejecuta el análisis completo de las cinco salidas obligatorias del proyecto."""
    warnings = []
    num_samples = len(iq)

    if fs_hz is None:
        warnings.append("Frecuencia de muestreo no especificada; los resultados espectrales se expresan en ciclos/muestra.")
        fs_for_spectrum = None
    else:
        fs_for_spectrum = float(fs_hz)

    # 1. Detección de presencia
    # Analizamos por ventanas de 256 muestras para consistencia con E1
    window_len = 256
    if num_samples >= window_len:
        eval_window = iq[:window_len]
    else:
        eval_window = iq
        warnings.append(f"Señal corta ({num_samples} muestras < 256); detección sobre longitud real.")

    power_obs = energy_statistic(eval_window)
    if detection_calibration is not None:
        det_res = detect_energy(eval_window, detection_calibration)
        is_detected = det_res["detected"]
        thresh_used = det_res["threshold"]
    else:
        thresh_used = float(default_threshold)
        is_detected = bool(power_obs > thresh_used)

    detection_out = {
        "detected": is_detected,
        "observed_energy": float(power_obs),
        "threshold": thresh_used,
        "status": "signal_present" if is_detected else "noise_only_or_undetected",
    }

    # 2. Espectro y Ancho de banda (PSD)
    # Estimamos sobre hasta 4096 muestras representativas
    spec_window = iq[: min(num_samples, 4096)]
    spec_res = estimate_spectrum(spec_window, fs_hz=fs_for_spectrum)
    bw_val = float(spec_res["total_power_bandwidth"])
    bw_unit = "Hz" if fs_for_spectrum is not None else "ciclos/muestra"
    freq_bounds = [float(spec_res["total_power_bounds"][0]), float(spec_res["total_power_bounds"][1])]

    spectrum_out = {
        "bandwidth_99": bw_val,
        "bandwidth_unit": bw_unit,
        "frequency_bounds": freq_bounds,
        "integrated_power": float(spec_res["integrated_power"]),
        "window_method": spec_res["window"],
        "spectrum_file": str(plot_path) if plot_path else None,
    }

    # 3. Estimación de SNR condicionada
    snr_raw = estimate_snr(eval_window, reference=noise_reference, detected=is_detected)
    if noise_reference is None:
        snr_out = {
            "snr_est_db": None,
            "status": "no_identificable_sin_supuestos",
            "reason": "falta_referencia_de_piso_de_ruido",
            "explanation": "La estimación rigurosa de SNR requiere una referencia independiente de ruido en la misma banda y ganancia.",
            "observed_power": float(power_obs),
        }
        warnings.append("SNR indeterminada: se requiere referencia externa de ruido estacionario.")
    else:
        snr_out = {
            "snr_est_db": float(snr_raw["snr_db"]) if snr_raw.get("snr_db") is not None else None,
            "status": snr_raw.get("status", "indeterminada"),
            "noise_power_estimate": snr_raw.get("noise_power_estimate"),
            "signal_power_estimate": snr_raw.get("signal_power_estimate"),
        }

    # 4 y 5. Clasificación y Confianza
    # Cargamos modelo convencional por defecto o especificado
    m_path = Path(model_path) if model_path else DEFAULT_SVM_MODEL
    classification_out = {
        "modulation": None,
        "confidence": None,
        "confidence_calibrated": False,
        "model_used": None,
        "scope": "signal_present_closed_set",
        "window_votes": {},
    }

    if not is_detected:
        classification_out["modulation"] = "NO_SIGNAL"
        classification_out["confidence"] = 0.0
        warnings.append("Detector negativo: modulación marcada como NO_SIGNAL.")
    elif not m_path.is_file():
        warnings.append(f"Modelo clasificador no encontrado en {m_path}; clasificación omitida.")
    else:
        # Cargar pipeline de SVM
        bundle = joblib.load(m_path)
        pipe = bundle["pipeline"]
        classes = list(bundle.get("classes", CLASSES))
        classification_out["model_used"] = str(m_path)

        # Cargar calibrador si está disponible
        calibrator = None
        p_path = Path(policy_path) if policy_path else DEFAULT_POLICY
        if p_path.is_file():
            try:
                pol = joblib.load(p_path)
                calibrator = pol.get("calibrator")
            except Exception:
                calibrator = None

        # Segmentar la señal en ventanas de 256 muestras
        n_windows = max(1, num_samples // window_len)
        window_preds = []
        window_margins = []

        for i in range(n_windows):
            start = i * window_len
            w = iq[start : start + window_len]
            if len(w) < window_len:
                break
            # Predict
            pred_idx = int(pipe.predict(w[None, :])[0])
            window_preds.append(classes[pred_idx])

            # Margen entre los dos mejores puntajes
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(w[None, :])[0]
                s_sorted = np.sort(scores)[::-1]
                margin = float(s_sorted[0] - s_sorted[1])
                window_margins.append(margin)

        # Votación mayoritaria
        if window_preds:
            unique_classes, counts = np.unique(window_preds, return_counts=True)
            best_class = unique_classes[np.argmax(counts)]
            vote_dict = {cls: int(cnt) for cls, cnt in zip(unique_classes, counts)}
            classification_out["modulation"] = str(best_class)
            classification_out["window_votes"] = vote_dict

            # Calibración de confianza
            if window_margins:
                mean_margin = float(np.mean(window_margins))
                if calibrator is not None:
                    # Usar regresión logística ajustada en val_cal (Fase Adaptativa)
                    prob_cal = float(calibrator.predict_proba([[mean_margin]])[0, 1])
                    # Escalar al rango de confianza multinomial
                    confidence_val = min(0.99, max(0.20, prob_cal * (counts.max() / len(window_preds))))
                    classification_out["confidence"] = round(confidence_val, 4)
                    classification_out["confidence_calibrated"] = True
                else:
                    # Sigmoide empírica sobre margen
                    confidence_val = float(1.0 / (1.0 + np.exp(-0.8 * mean_margin)))
                    classification_out["confidence"] = round(min(0.99, max(0.20, confidence_val)), 4)
                    classification_out["confidence_calibrated"] = False
        else:
            classification_out["modulation"] = "INDETERMINATE"
            classification_out["confidence"] = 0.0

    # 6. Generación del gráfico si fue solicitado
    if plot_path:
        plot_p = Path(plot_path)
        plot_p.parent.mkdir(parents=True, exist_ok=True)
        generate_report_plot(
            iq=iq,
            spec_res=spec_res,
            output_png=plot_p,
            fs_hz=fs_for_spectrum,
            title=f"Reporte de Señal: {signal_id} ({classification_out['modulation']})",
        )

    return {
        "signal_id": signal_id,
        "num_samples": num_samples,
        "fs_hz": fs_hz,
        "detection": detection_out,
        "spectrum": spectrum_out,
        "snr": snr_out,
        "classification": classification_out,
        "warnings": warnings,
    }


def generate_report_plot(iq, spec_res, output_png, fs_hz=8000.0, title=None):
    """Genera gráfico de reporte en PNG con densidad espectral (PSD) y constelación."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel izquierdo: Densidad espectral de potencia (PSD)
    freqs = spec_res["frequency"]
    psd_db = 10 * np.log10(np.maximum(spec_res["psd"], 1e-12))
    axes[0].plot(freqs, psd_db, color="royalblue", lw=1.5, label="PSD Observada")

    # Sombrear banda al 99%
    b_min, b_max = spec_res["total_power_bounds"]
    mask = (freqs >= b_min) & (freqs <= b_max)
    axes[0].fill_between(freqs[mask], psd_db[mask], -100, color="lightblue", alpha=0.5, label="Banda 99%")

    axes[0].axvline(b_min, color="crimson", linestyle="--", alpha=0.7, label=f"Límite: {b_min:.1f}")
    axes[0].axvline(b_max, color="crimson", linestyle="--", alpha=0.7, label=f"Límite: {b_max:.1f}")

    x_unit = "Hz" if fs_hz is not None else "ciclos/muestra"
    axes[0].set_xlabel(f"Frecuencia ({x_unit})")
    axes[0].set_ylabel("PSD (dB/Hz)")
    axes[0].set_title(f"Densidad Espectral (BW 99% = {spec_res['total_power_bandwidth']:.1f} {x_unit})")
    axes[0].set_ylim([max(-90, float(psd_db.min()) - 5), max(10, float(psd_db.max()) + 5)])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel derecho: Diagrama de constelación
    sub_iq = iq[: min(len(iq), 1024)]
    axes[1].scatter(np.real(sub_iq), np.imag(sub_iq), alpha=0.5, color="darkorange", s=14, edgecolors="none")
    axes[1].set_xlabel("En-Fase (I)")
    axes[1].set_ylabel("Cuadratura (Q)")
    axes[1].set_title(f"Diagrama de Constelación ({len(sub_iq)} muestras)")
    max_val = max(1.5, float(np.max(np.abs(sub_iq))) * 1.1)
    axes[1].set_xlim([-max_val, max_val])
    axes[1].set_ylim([-max_val, max_val])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].axhline(0, color="gray", lw=0.8, alpha=0.5)
    axes[1].axvline(0, color="gray", lw=0.8, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_png), dpi=160)
    plt.close()


def main():
    """Punto de entrada CLI para el monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor Integrado de Cinco Salidas para Señales Desconocidas (COM2)"
    )
    parser.add_argument("--input", "-i", required=True, help="Ruta al archivo de señal (.c64, .bin, .npy, .npz, .csv)")
    parser.add_argument("--fs-hz", type=float, default=8000.0, help="Frecuencia de muestreo en Hz (por defecto 8000)")
    parser.add_argument("--dtype", default="complex64", help="Tipo de dato binario (por defecto complex64)")
    parser.add_argument("--model", "-m", default=str(DEFAULT_SVM_MODEL), help="Ruta al modelo clasificador .joblib")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="Ruta al calibrador / política .joblib")
    parser.add_argument("--output", "-o", default=None, help="Ruta al archivo JSON de resultados")
    parser.add_argument("--plot", "-p", default=None, help="Ruta a la figura PNG de espectro y constelación")
    parser.add_argument("--noise-ref", default=None, help="Ruta a archivo de ruido de referencia para estimar SNR")
    parser.add_argument("--threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD, help="Umbral de energía del detector")

    args = parser.parse_args()

    input_path = Path(args.input)
    signal_id = input_path.stem

    # Determinar rutas de salida por defecto si no fueron dadas
    results_dir = Path("results/monitor")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else results_dir / f"{signal_id}.json"
    out_plot = Path(args.plot) if args.plot else results_dir / f"{signal_id}_psd.png"

    # Cargar referencia de ruido si existe
    noise_ref = None
    if args.noise_ref:
        noise_samples = load_signal(args.noise_ref, dtype=args.dtype)
        noise_ref = calibrate_noise_reference(noise_samples)

    # Cargar señal
    iq = load_signal(input_path, dtype=args.dtype)

    # Ejecutar análisis
    res = analyze_complete(
        iq=iq,
        signal_id=signal_id,
        fs_hz=args.fs_hz,
        model_path=args.model,
        policy_path=args.policy,
        noise_reference=noise_ref,
        default_threshold=args.threshold,
        plot_path=out_plot,
    )

    # Guardar JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f" MONITOR INTEGRADO COM2 - REPORTE DE SEÑAL [{signal_id}]")
    print("=" * 60)
    print(f"  Muestras totales   : {res['num_samples']}")
    print(f"  Frecuencia muestreo: {res['fs_hz']} Hz")
    print(f"  1. Detección       : {'SÍ (Señal presente)' if res['detection']['detected'] else 'NO (Solo ruido)'}")
    print(f"     Energía observad: {res['detection']['observed_energy']:.4f} (Umbral: {res['detection']['threshold']:.4f})")
    print(f"  2. Ancho de Banda  : {res['spectrum']['bandwidth_99']:.2f} {res['spectrum']['bandwidth_unit']} (99% potencia)")
    print(f"     Banda ocupada   : [{res['spectrum']['frequency_bounds'][0]:.1f}, {res['spectrum']['frequency_bounds'][1]:.1f}]")
    print(f"  3. SNR Estimada    : {res['snr']['snr_est_db']} dB ({res['snr']['status']})")
    print(f"  4. Modulación      : {res['classification']['modulation']}")
    print(f"  5. Confianza       : {res['classification']['confidence'] * 100:.1f}% (Calibrada: {res['classification']['confidence_calibrated']})")
    print("-" * 60)
    print(f"  Archivo JSON       : {out_json}")
    print(f"  Gráfico PSD / Const: {out_plot}")
    if res["warnings"]:
        print("  Advertencias:")
        for w in res["warnings"]:
            print(f"    * {w}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
