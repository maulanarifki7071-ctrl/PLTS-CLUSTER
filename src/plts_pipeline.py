from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Safer CPU behavior on Windows laptops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# General utilities
# =============================================================================

def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(str, parts))


def run_dir(config: Dict[str, Any], run_name: str | None = None) -> Path:
    if not run_name:
        run_name = config["project"].get("default_run_name", "final_run_v3")
    return ensure_dir(project_path(config["paths"]["output_dir"], run_name))


def raw_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(project_path(config["paths"]["raw_dir"]))


def preprocessed_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(project_path(config["paths"]["preprocessed_dir"]))


def processed_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(project_path(config["paths"]["processed_dir"]))


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or math.isclose(float(mx), float(mn)):
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - mn) / (mx - mn)
    return 1 - out if invert else out


def coefficient_of_variation(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    mean = s.mean()
    if np.isclose(mean, 0):
        return np.nan
    return float(s.std(ddof=1) / mean)


def numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame(columns=["column", "min", "max", "mean", "std"])
    return num.agg(["min", "max", "mean", "std"]).T.reset_index().rename(columns={"index": "column"})


def normalize_string(x: Any) -> str:
    return str(x).strip().lower()

def cleanup_essential_outputs(config: Dict[str, Any], rd: Path) -> None:
    """Keep only report-facing outputs when outputs.mode is essential_report."""
    if config.get("outputs", {}).get("mode") != "essential_report":
        return
    essential = {
        "00_raw_audit": {
            "final_dataset_summary.txt",
            "final_dataset_validation.csv",
            "final_dataset_validation.json",
            "raw_audit_summary.csv",
            "raw_audit_summary.json",
            "raw_per_year_summary.csv",
        },
        "01_preprocessing_audit": {
            "preprocessing_audit_summary.csv",
            "preprocessing_audit_summary.json",
            "changed_island_group_rows.csv",
            "corrected_island_group_distribution.csv",
        },
        "02_summary_audit": {
            "summary_audit.csv",
            "summary_audit.json",
            "feature_retention_decision.csv",
            "core_context_6_correlation_pairs.csv",
            "summary_island_group_distribution.csv",
        },
        "03_clustering": {
            "model_comparison_metrics.csv",
            "primary_scenario_metrics.csv",
            "selected_final_model_summary.csv",
            "selected_model_summary.json",
            "metric_best_primary_model_summary.csv",
            "k2_vs_k3_tradeoff.csv",
            "feature_sensitivity_comparison.csv",
            "stability_subsampling_ari.csv",
            "clustering_results.csv",  # required by step 04 visual generation
            "cluster_profile.csv",
            "cluster_interpretation.csv",
            "cluster_island_group_distribution.csv",
            "cluster_label_mapping.csv",
        },
        "04_figures": {
            "comparison_silhouette_score.png",
            "comparison_davies_bouldin_index.png",
            "comparison_calinski_harabasz_score.png",
            "comparison_balance_score.png",
            "comparison_stability_ari.png",
            "comparison_composite_score_final.png",
            "pca_cluster_scatter.png",
            "cluster_profile_heatmap_z.png",
            "cluster_profile_radar.png",
            "cluster_island_group_distribution.png",
        },
        "06_reports": {
            "final_clustering_report.md",
            "README_OUTPUT.md",
        },
    }
    for folder, keep_names in essential.items():
        d = rd / folder
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.is_file():
                rel_parts = path.relative_to(d).parts
                if len(rel_parts) != 1 or path.name not in keep_names:
                    path.unlink(missing_ok=True)
    # Remove non-essential maps folder and empty subdirectories created by optional outputs.
    maps_dir = rd / "05_maps"
    if maps_dir.exists():
        for path in sorted(maps_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        maps_dir.rmdir()
    for path in sorted(rd.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


# =============================================================================
# Step 00 - Raw audit
# =============================================================================

def read_raw_files(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    infos: List[Dict[str, Any]] = []
    rdir = raw_dir(config)
    for filename in config["raw_files"]:
        path = rdir / filename
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        df = pd.read_csv(path)
        frames.append(df)
        years = sorted(pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
        infos.append({
            "file": filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "years_detected": ",".join(map(str, years)),
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(infos)


def audit_raw(config: Dict[str, Any], run_name: str | None = None) -> None:
    rd = run_dir(config, run_name)
    out = ensure_dir(rd / "00_raw_audit")
    df, info = read_raw_files(config)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_regions": int(df["region_id"].nunique()),
        "unique_dates": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_region_date": int(df.duplicated(["region_id", "date"]).sum()),
        "missing_total": int(df.isna().sum().sum()),
    }

    per_year = (
        df.groupby("year")
        .agg(
            rows=("region_id", "size"),
            unique_regions=("region_id", "nunique"),
            unique_dates=("date", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )
    per_year["date_min"] = per_year["date_min"].dt.date.astype(str)
    per_year["date_max"] = per_year["date_max"].dt.date.astype(str)

    region_day_count = df.groupby("region_id")["date"].nunique().reset_index(name="n_days")
    expected_days_total = int(config["preprocessing"].get("expected_days_total", 1096))
    initial_target = int(config.get("sampling", {}).get("initial_target_regions", 150))
    final_expected = int(config.get("sampling", {}).get("final_valid_regions_expected", summary["unique_regions"]))
    region_day_count["completeness_ratio"] = region_day_count["n_days"] / expected_days_total
    region_day_count["is_complete_expected_period"] = region_day_count["n_days"] == expected_days_total
    expected_rows = int(summary["unique_regions"]) * expected_days_total
    validation = {
        "initial_target_regions_from_gee_script": initial_target,
        "final_valid_regions_detected": int(summary["unique_regions"]),
        "final_valid_regions_expected_config": final_expected,
        "expected_days_total": expected_days_total,
        "expected_rows_for_detected_regions": expected_rows,
        "actual_rows": int(summary["rows"]),
        "row_count_matches_detected_regions_x_days": bool(int(summary["rows"]) == expected_rows),
        "all_regions_complete_expected_days": bool(region_day_count["is_complete_expected_period"].all()),
        "duplicate_region_date": int(summary["duplicate_region_date"]),
        "missing_total": int(summary["missing_total"]),
        "dataset_status_note": "Dataset final yang dianalisis adalah wilayah valid lengkap hasil ekspor GEE. Target awal 150 titik tidak diklaim sebagai jumlah final bila audit mendeteksi 137 wilayah valid."
    }

    save_csv(info, out / "raw_file_info.csv")
    save_csv(pd.DataFrame([summary]), out / "raw_audit_summary.csv")
    write_json(summary, out / "raw_audit_summary.json")
    save_csv(per_year, out / "raw_per_year_summary.csv")
    save_csv(pd.DataFrame([validation]), out / "final_dataset_validation.csv")
    write_json(validation, out / "final_dataset_validation.json")
    with (out / "final_dataset_summary.txt").open("w", encoding="utf-8") as f:
        f.write("RINGKASAN DATASET FINAL\n")
        f.write("======================\n")
        for key, value in validation.items():
            f.write(f"{key}: {value}\n")
    save_csv(region_day_count.sort_values("completeness_ratio"), out / "raw_region_day_completeness.csv")
    save_csv(df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"}), out / "raw_missing_values.csv")
    save_csv(numeric_ranges(df), out / "raw_variable_ranges.csv")

    cleanup_essential_outputs(config, rd)
    print("STEP 00 selesai. Raw audit:", summary)


# =============================================================================
# Step 01 - Preprocessing daily data
# =============================================================================

def coordinate_group(lon: float, lat: float) -> str:
    """Fallback island group from centroid coordinates.

    This is used only when region name overrides are absent. Ordering avoids the
    original overlap problem where western Java could be captured as Sumatra.
    """
    if 114 <= lon < 126 and -12 <= lat < -7:
        return "Bali_Nusa_Tenggara"
    if 105.5 <= lon < 115.8 and -10 <= lat < -5.0:
        return "Jawa"
    if 118 <= lon < 126 and -7 <= lat <= 3:
        return "Sulawesi"
    if 108 <= lon < 120 and -5 <= lat <= 5:
        return "Kalimantan"
    if 131 <= lon <= 142 and -10 <= lat <= 2:
        return "Papua"
    if 124 <= lon < 132 and -9 <= lat <= 3:
        return "Maluku"
    if 94 <= lon < 107 and -7 <= lat <= 7:
        return "Sumatra"
    return "Other"


def correct_island_group(row: pd.Series, config: Dict[str, Any]) -> str:
    overrides = config.get("preprocessing", {}).get("island_group_overrides", {}) or {}
    lower_overrides = {normalize_string(k): str(v) for k, v in overrides.items()}
    name = normalize_string(row.get("region_name", ""))
    if name in lower_overrides:
        return lower_overrides[name]
    return coordinate_group(float(row["centroid_longitude"]), float(row["centroid_latitude"]))


def preprocess(config: Dict[str, Any], run_name: str | None = None) -> None:
    rd = run_dir(config, run_name)
    audit_out = ensure_dir(rd / "01_preprocessing_audit")
    out_dir = preprocessed_dir(config)

    df, _ = read_raw_files(config)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["island_group_original"] = df["island_group"].astype(str)
    meta_cols = ["region_id", "region_name", "island_group_original", "centroid_longitude", "centroid_latitude"]
    meta = df[meta_cols].drop_duplicates("region_id").copy()
    meta["island_group_corrected"] = meta.apply(lambda r: correct_island_group(r, config), axis=1)
    meta["changed"] = meta["island_group_original"] != meta["island_group_corrected"]
    df = df.merge(meta[["region_id", "island_group_corrected"]], on="region_id", how="left")

    neg_before = int((df["precipitation_mm"] < 0).sum())
    min_precip_before = float(df["precipitation_mm"].min())
    if config.get("preprocessing", {}).get("clip_negative_precipitation", True):
        df.loc[df["precipitation_mm"] < 0, "precipitation_mm"] = 0.0

    hot_threshold = float(config["thresholds"]["hot_temp_c"])
    thermal_base = float(config["thresholds"]["thermal_base_temp_c"])
    rainy_threshold = float(config["thresholds"]["rainy_day_mm"])

    df["is_rainy_day"] = (df["precipitation_mm"] >= rainy_threshold).astype(int)
    df["hot_day_35c"] = (df["daily_max_temperature_2m_c"] >= hot_threshold).astype(int)
    df["thermal_heat_excess_c"] = np.maximum(df["daily_max_temperature_2m_c"] - thermal_base, 0.0)
    df["thermal_risk_proxy"] = df["solar_irradiance_kwh_m2"] * df["thermal_heat_excess_c"]
    df["hot_irradiance_35c_interaction"] = df["solar_irradiance_kwh_m2"] * df["hot_day_35c"]

    df["solar_irradiance_kwh_m2_check"] = df["solar_radiation_j_m2"] / 3_600_000.0
    df["solar_irradiance_abs_diff"] = (df["solar_irradiance_kwh_m2"] - df["solar_irradiance_kwh_m2_check"]).abs()

    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    clean_path = out_dir / config["files"]["combined_daily_clean"]
    save_csv(df, clean_path)
    save_csv(meta.sort_values(["changed", "island_group_corrected", "region_name"], ascending=[False, True, True]), audit_out / "island_group_correction_audit.csv")
    save_csv(meta[meta["changed"]].sort_values(["island_group_corrected", "region_name"]), audit_out / "changed_island_group_rows.csv")
    save_csv(df.drop_duplicates("region_id").groupby("island_group_corrected").size().reset_index(name="n_regions"), audit_out / "corrected_island_group_distribution.csv")

    audit = {
        "output_clean_daily": str(clean_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(df)),
        "unique_regions": int(df["region_id"].nunique()),
        "unique_dates": int(pd.to_datetime(df["date"]).nunique()),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_region_date": int(df.duplicated(["region_id", "date"]).sum()),
        "island_group_changed_regions": int(meta["changed"].sum()),
        "precipitation_negative_before_clip": neg_before,
        "precipitation_min_before_clip": min_precip_before,
        "max_conversion_diff_kwh_m2": float(df["solar_irradiance_abs_diff"].max()),
    }
    save_csv(pd.DataFrame([audit]), audit_out / "preprocessing_audit_summary.csv")
    write_json(audit, audit_out / "preprocessing_audit_summary.json")

    cleanup_essential_outputs(config, rd)
    print("STEP 01 selesai. Clean daily saved to:", clean_path)
    print("Audit:", audit)


# =============================================================================
# Step 02 - Summary + EDA audit
# =============================================================================

def add_indicative_context_score(summary: pd.DataFrame) -> pd.DataFrame:
    """Indicative exploratory score, not final feasibility/economic score."""
    s = summary.copy()
    s["score_irradiance"] = minmax(s["mean_daily_irradiance_kwh_m2"])
    s["score_stability"] = minmax(s["monthly_irradiance_cv"], invert=True)
    s["score_low_rain"] = minmax(s["rainy_day_ratio"], invert=True)
    s["score_low_humidity"] = minmax(s["relative_humidity_mean_pct"], invert=True)
    # Use heat excess rather than thermal_risk_proxy to avoid double-counting irradiance in the indicative score.
    s["score_low_thermal"] = minmax(s["thermal_heat_excess_mean_c"], invert=True)
    s["indicative_context_score"] = (
        0.40 * s["score_irradiance"]
        + 0.20 * s["score_stability"]
        + 0.15 * s["score_low_rain"]
        + 0.10 * s["score_low_humidity"]
        + 0.15 * s["score_low_thermal"]
    )
    s["indicative_context_rank"] = s["indicative_context_score"].rank(ascending=False, method="min").astype(int)
    return s


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].replace([np.inf, -np.inf], np.nan).copy()
    X = X.fillna(X.median(numeric_only=True))
    X = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    X = X.fillna(0.0).to_numpy(dtype=float)
    rows = []
    for i, feature in enumerate(features):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0 or np.allclose(y.var(), 0):
            vif = np.nan
        else:
            beta, *_ = np.linalg.lstsq(others, y, rcond=None)
            y_hat = others @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = np.inf if r2 >= 0.999999 else 1 / max(1 - r2, 1e-12)
        rows.append({"feature": feature, "vif": float(vif) if np.isfinite(vif) else np.inf})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def build_summary(config: Dict[str, Any], run_name: str | None = None) -> None:
    rd = run_dir(config, run_name)
    audit_out = ensure_dir(rd / "02_summary_audit")
    pdir = processed_dir(config)
    pre_path = preprocessed_dir(config) / config["files"]["combined_daily_clean"]
    if not pre_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found. Run step 01 first: {pre_path}")

    df = pd.read_csv(pre_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    g = df.groupby("region_id")

    ident_cols = [
        "region_id", "region_name", "island_group_original", "island_group_corrected",
        "centroid_longitude", "centroid_latitude", "boundary_source", "sampling_method",
        "data_source_boundary", "data_source_weather", "data_source_elevation",
    ]
    ident = df[[c for c in ident_cols if c in df.columns]].drop_duplicates("region_id").copy()

    summary = pd.DataFrame(index=sorted(df["region_id"].unique()))
    summary.index.name = "region_id"
    summary["n_days"] = g["date"].nunique()
    summary["start_date"] = g["date"].min().dt.strftime("%Y-%m-%d")
    summary["end_date"] = g["date"].max().dt.strftime("%Y-%m-%d")

    # Radiation.
    summary["mean_daily_irradiance_kwh_m2"] = g["solar_irradiance_kwh_m2"].mean()
    summary["median_daily_irradiance_kwh_m2"] = g["solar_irradiance_kwh_m2"].median()
    summary["irradiance_std"] = g["solar_irradiance_kwh_m2"].std(ddof=1)
    summary["irradiance_cv"] = g["solar_irradiance_kwh_m2"].apply(coefficient_of_variation)

    monthly = df.groupby(["region_id", "year_month"])["solar_irradiance_kwh_m2"].mean().reset_index(name="monthly_mean_irradiance")
    mg = monthly.groupby("region_id")["monthly_mean_irradiance"]
    summary["monthly_irradiance_mean"] = mg.mean()
    summary["monthly_irradiance_std"] = mg.std(ddof=1)
    summary["monthly_irradiance_cv"] = mg.apply(coefficient_of_variation)

    # Temperature.
    summary["temperature_max_c"] = g["daily_max_temperature_2m_c"].max()
    summary["temperature_p90_c"] = g["daily_max_temperature_2m_c"].quantile(0.90)
    summary["temperature_daily_max_mean_c"] = g["daily_max_temperature_2m_c"].mean()
    summary["daily_mean_temperature_mean_c"] = g["daily_mean_temperature_2m_c"].mean()
    summary["temp_dewpoint_spread_mean_c"] = g["temp_dewpoint_spread_c"].mean()

    # Thermal/extreme heat.
    summary["hot_day_35c_ratio"] = g["hot_day_35c"].mean()
    summary["hot_day_35c_count"] = g["hot_day_35c"].sum()
    summary["thermal_heat_excess_mean_c"] = g["thermal_heat_excess_c"].mean()
    summary["thermal_risk_proxy_mean"] = g["thermal_risk_proxy"].mean()
    summary["thermal_risk_proxy_p90"] = g["thermal_risk_proxy"].quantile(0.90)
    summary["hot_irradiance_35c_interaction_mean"] = g["hot_irradiance_35c_interaction"].mean()

    # Rain/humidity/wind/elevation.
    summary["precipitation_mean_mm"] = g["precipitation_mm"].mean()
    summary["precipitation_total_mm"] = g["precipitation_mm"].sum()
    summary["precipitation_p90_mm"] = g["precipitation_mm"].quantile(0.90)
    summary["rainy_day_ratio"] = g["is_rainy_day"].mean()
    summary["rainy_day_count"] = g["is_rainy_day"].sum()
    summary["relative_humidity_mean_pct"] = g["relative_humidity_2m_pct"].mean()
    summary["relative_humidity_p90_pct"] = g["relative_humidity_2m_pct"].quantile(0.90)
    summary["wind_speed_mean_ms"] = g["wind_speed_10m_ms"].mean()
    summary["wind_speed_p90_ms"] = g["wind_speed_10m_ms"].quantile(0.90)
    summary["elevation_m"] = g["elevation_m"].median()

    summary = ident.merge(summary.reset_index(), on="region_id", how="right")
    summary = add_indicative_context_score(summary)
    summary = summary.sort_values("indicative_context_rank").reset_index(drop=True)

    out_path = pdir / config["files"]["summary_output"]
    save_csv(summary, out_path)

    feature_definitions = pd.DataFrame([
        {"feature": "mean_daily_irradiance_kwh_m2", "category": "radiasi", "definition": "Rata-rata radiasi matahari harian hasil konversi J/m2 ke kWh/m2; indikator utama potensi sumber energi surya."},
        {"feature": "monthly_irradiance_cv", "category": "stabilitas radiasi", "definition": "Koefisien variasi rata-rata radiasi bulanan; makin rendah berarti pola radiasi bulanan lebih stabil."},
        {"feature": "temperature_p90_c", "category": "suhu", "definition": "Persentil ke-90 suhu maksimum harian; indikator suhu tinggi yang lebih stabil daripada threshold ekstrem."},
        {"feature": "thermal_heat_excess_mean_c", "category": "termal-diagnostik", "definition": "Rata-rata max(Tmax - 30, 0); indikator diagnostik tekanan panas di atas suhu dasar 30 C, tidak digunakan sebagai fitur utama pada revisi 6 variabel."},
        {"feature": "thermal_risk_proxy_mean", "category": "termal-interaksi", "definition": "Rata-rata irradiance x thermal_heat_excess; fitur interaksi radiasi dan tekanan panas, bukan suhu panel aktual dan bukan rugi-rugi efisiensi aktual."},
        {"feature": "hot_day_35c_ratio", "category": "suhu ekstrem-diagnostik", "definition": "Proporsi hari dengan suhu maksimum harian >= 35 C; indikator diagnostik hari panas ekstrem, bukan fitur utama clustering."},
        {"feature": "rainy_day_ratio", "category": "hujan", "definition": "Proporsi hari dengan curah hujan >= 1 mm; indikator frekuensi hari hujan."},
        {"feature": "relative_humidity_mean_pct", "category": "kelembapan", "definition": "Rata-rata kelembapan relatif hasil turunan pendekatan Magnus dari suhu dan dew point, dibatasi 0-100 persen."},
        {"feature": "elevation_m", "category": "topografi", "definition": "Elevasi median pada titik centroid dari SRTM; dipakai sebagai konteks topografi, bukan detail atap."},
    ])
    save_csv(feature_definitions, pdir / "feature_definition_table.csv")

    audit_feature_set = config.get("clustering", {}).get("primary_feature_set", "thermal_alternative")
    feature_sets_cfg = config.get("clustering", {}).get("feature_sets", {})
    features = list(feature_sets_cfg.get(audit_feature_set, {}).get("features", config["feature_engineering"]["balanced_context_features"]))
    corr = summary[features].corr(numeric_only=True)
    save_csv(corr.reset_index().rename(columns={"index": "feature"}), audit_out / f"{audit_feature_set}_correlation_matrix.csv")
    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            r = corr.loc[f1, f2]
            pairs.append({"feature_1": f1, "feature_2": f2, "correlation": r, "abs_correlation": abs(r), "flag_gt_threshold": abs(r) > float(config["feature_engineering"].get("correlation_threshold", 0.85))})
    corr_pairs_df = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False)
    save_csv(corr_pairs_df, audit_out / f"{audit_feature_set}_correlation_pairs.csv")
    vif_df = compute_vif(summary, features)
    save_csv(vif_df, audit_out / f"{audit_feature_set}_vif.csv")

    # Transparent feature-retention note for thesis methodology.
    # Some features can be correlated, but they are retained when their physical
    # meaning differs in PLTS context. This prevents the report from claiming
    # that all correlated features were automatically removed.
    retention_notes = []
    high_pairs = corr_pairs_df[corr_pairs_df["flag_gt_threshold"] == True]
    for _, r in high_pairs.iterrows():
        f1, f2 = str(r["feature_1"]), str(r["feature_2"])
        if {f1, f2} == {"temperature_p90_c", "elevation_m"}:
            decision = "retained"
            reason = "temperature_p90_c represents heat condition, while elevation_m represents topographic context; both are physically different in PLTS interpretation."
        elif {f1, f2} == {"rainy_day_ratio", "relative_humidity_mean_pct"}:
            decision = "retained"
            reason = "rainy_day_ratio represents rainy-day frequency, while relative_humidity_mean_pct represents atmospheric moisture; both are interpreted carefully as related but not identical context variables."
        else:
            decision = "retained_with_caution"
            reason = "Correlation exceeds threshold, but feature retention is evaluated with domain relevance and interpretability, not removed automatically."
        retention_notes.append({
            "feature_1": f1,
            "feature_2": f2,
            "correlation": r["correlation"],
            "abs_correlation": r["abs_correlation"],
            "decision": decision,
            "reason": reason,
        })
    if not retention_notes:
        retention_notes.append({"feature_1": "-", "feature_2": "-", "correlation": np.nan, "abs_correlation": np.nan, "decision": "no_high_correlation_flag", "reason": "No pair exceeded the configured correlation threshold."})
    save_csv(pd.DataFrame(retention_notes), audit_out / "feature_retention_decision.csv")

    z = summary[features].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    outlier_rows = []
    z_thr = float(config["feature_engineering"].get("outlier_z_threshold", 3.0))
    for feature in features:
        flagged = z[feature].abs() > z_thr
        if flagged.any():
            temp = summary.loc[flagged, ["region_id", "region_name", "island_group_corrected"]].copy()
            temp["feature"] = feature
            temp["value"] = summary.loc[flagged, feature].values
            temp["z_score"] = z.loc[flagged, feature].values
            outlier_rows.append(temp)
    if outlier_rows:
        save_csv(pd.concat(outlier_rows, ignore_index=True), audit_out / f"{audit_feature_set}_outlier_candidates.csv")
    else:
        save_csv(pd.DataFrame(columns=["region_id", "region_name", "island_group_corrected", "feature", "value", "z_score"]), audit_out / f"{audit_feature_set}_outlier_candidates.csv")

    island_dist = summary.groupby("island_group_corrected").size().reset_index(name="n_regions")
    save_csv(island_dist, audit_out / "summary_island_group_distribution.csv")

    # Centroid representativeness audit: this does not claim polygon-wide
    # representativeness; it documents the geographic and environmental spread
    # of the 137 centroid samples used by the study.
    centroid_audit_rows = []
    overall_cols = [
        "centroid_longitude", "centroid_latitude", "elevation_m",
        "mean_daily_irradiance_kwh_m2", "temperature_p90_c",
        "rainy_day_ratio", "relative_humidity_mean_pct"
    ]
    for group_name, part in [("ALL", summary)] + list(summary.groupby("island_group_corrected")):
        row = {"scope": group_name, "n_regions": int(len(part))}
        for col in overall_cols:
            if col in part.columns:
                row[f"{col}_min"] = float(part[col].min())
                row[f"{col}_max"] = float(part[col].max())
                row[f"{col}_mean"] = float(part[col].mean())
        centroid_audit_rows.append(row)
    save_csv(pd.DataFrame(centroid_audit_rows), audit_out / "centroid_representativeness_audit.csv")

    save_csv(summary.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"}), audit_out / "summary_missing_values.csv")
    save_csv(numeric_ranges(summary), audit_out / "summary_variable_ranges.csv")

    audit = {
        "output_summary": str(out_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(summary)),
        "columns": int(len(summary.columns)),
        "missing_total": int(summary.isna().sum().sum()),
        "min_n_days": int(summary["n_days"].min()),
        "max_n_days": int(summary["n_days"].max()),
        "feature_audit_set": audit_feature_set,
        "feature_audit_features": features,
    }
    save_csv(pd.DataFrame([audit]), audit_out / "summary_audit.csv")
    write_json(audit, audit_out / "summary_audit.json")

    cleanup_essential_outputs(config, rd)
    print("STEP 02 selesai. Summary saved to:", out_path)
    print("Audit:", audit)


# =============================================================================
# Step 03 - Clustering final revision
# =============================================================================

def make_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "power_yj":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unknown scaler: {name}")


def prepare_matrix(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[features].replace([np.inf, -np.inf], np.nan).copy()
    rows = []
    for col in X.columns:
        miss = int(X[col].isna().sum())
        med = float(X[col].median()) if X[col].notna().any() else 0.0
        if miss:
            X[col] = X[col].fillna(med)
        rows.append({"feature": col, "missing_count": miss, "impute_value": med})
    return X, pd.DataFrame(rows)


def model_params(config: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    """Read model hyperparameters from the final structured config.

    The fallback keys keep the script compatible with older config versions, but
    the recommended final config uses `clustering.model_params.<algorithm>`.
    """
    cfg = config.get("clustering", {})
    mp = cfg.get("model_params", {}) or {}
    algorithm = algorithm.lower()
    params = dict(mp.get(algorithm, {}) or {})
    if algorithm == "kmeans":
        params.setdefault("n_init", cfg.get("kmeans_n_init", 50))
        params.setdefault("max_iter", cfg.get("kmeans_max_iter", 500))
        params.setdefault("random_state", cfg.get("random_state", 2025))
    elif algorithm == "ahc":
        params.setdefault("linkage", cfg.get("ahc_linkage", "ward"))
        params.setdefault("metric", cfg.get("ahc_metric", "euclidean"))
    elif algorithm == "gmm":
        params.setdefault("n_init", cfg.get("gmm_n_init", 10))
        params.setdefault("max_iter", cfg.get("gmm_max_iter", 500))
        cov = cfg.get("gmm_covariance_type", "full")
        params.setdefault("covariance_types", cov if isinstance(cov, list) else [cov])
        params.setdefault("selection_criterion", cfg.get("gmm_selection_criterion", "bic"))
        params.setdefault("reg_covar", cfg.get("gmm_reg_covar", 1e-6))
        params.setdefault("random_state", cfg.get("random_state", 2025))
    return params


def make_ahc(n_clusters: int, linkage_name: str, metric_name: str):
    """Create AgglomerativeClustering while supporting old/new sklearn APIs."""
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_name, metric=metric_name)
    except TypeError:
        # Older scikit-learn versions used `affinity` instead of `metric`.
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_name, affinity=metric_name)


def fit_predict(algorithm: str, k: int, X_scaled: np.ndarray, config: Dict[str, Any], seed: int) -> Tuple[Any, np.ndarray, np.ndarray]:
    algorithm = algorithm.lower()
    params = model_params(config, algorithm)

    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k,
            random_state=int(params.get("random_state", seed)),
            n_init=int(params.get("n_init", 50)),
            max_iter=int(params.get("max_iter", 500)),
        )
        labels = model.fit_predict(X_scaled)
        confidence = np.full(len(labels), np.nan)

    elif algorithm == "ahc":
        linkage_name = str(params.get("linkage", "ward"))
        metric_name = str(params.get("metric", "euclidean"))
        model = make_ahc(k, linkage_name, metric_name)
        labels = model.fit_predict(X_scaled)
        model.selected_metric_ = metric_name
        confidence = np.full(len(labels), np.nan)

    elif algorithm == "gmm":
        cov_types = params.get("covariance_types", ["diag", "full"])
        if isinstance(cov_types, str):
            cov_types = [cov_types]
        criterion = str(params.get("selection_criterion", "bic")).lower()
        candidates = []
        for cov_type in cov_types:
            gm = GaussianMixture(
                n_components=k,
                random_state=int(params.get("random_state", seed)),
                n_init=int(params.get("n_init", 10)),
                max_iter=int(params.get("max_iter", 500)),
                covariance_type=str(cov_type),
                reg_covar=float(params.get("reg_covar", 1e-6)),
            )
            gm.fit(X_scaled)
            candidates.append({
                "model": gm,
                "covariance_type": str(cov_type),
                "bic": float(gm.bic(X_scaled)),
                "aic": float(gm.aic(X_scaled)),
            })
        key = "bic" if criterion not in {"aic"} else "aic"
        best = min(candidates, key=lambda item: item[key])
        model = best["model"]
        model.covariance_selection_candidates_ = candidates
        model.selected_covariance_type_ = best["covariance_type"]
        model.selected_bic_ = best["bic"]
        model.selected_aic_ = best["aic"]
        model.selection_criterion_ = criterion
        labels = model.predict(X_scaled)
        confidence = model.predict_proba(X_scaled).max(axis=1)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, labels.astype(int), confidence


def model_info_row(model: Any, algorithm: str) -> Dict[str, Any]:
    """Small metadata row for auditability of model hyperparameters used."""
    algorithm = algorithm.lower()
    if algorithm == "kmeans":
        return {
            "kmeans_inertia": float(getattr(model, "inertia_", np.nan)),
            "kmeans_n_iter": int(getattr(model, "n_iter_", -1)),
            "model_param_note": "K-Means uses n_init/max_iter/random_state from config model_params.",
        }
    if algorithm == "ahc":
        return {
            "ahc_linkage": getattr(model, "linkage", np.nan),
            "ahc_metric": getattr(model, "selected_metric_", getattr(model, "metric", np.nan)),
            "model_param_note": "AHC is deterministic for the same data/scaler/linkage/metric.",
        }
    if algorithm == "gmm":
        return {
            "gmm_selected_covariance_type": getattr(model, "selected_covariance_type_", getattr(model, "covariance_type", np.nan)),
            "gmm_selected_bic": float(getattr(model, "selected_bic_", np.nan)),
            "gmm_selected_aic": float(getattr(model, "selected_aic_", np.nan)),
            "gmm_selection_criterion": getattr(model, "selection_criterion_", np.nan),
            "gmm_converged": bool(getattr(model, "converged_", False)),
            "gmm_n_iter": int(getattr(model, "n_iter_", -1)),
            "model_param_note": "GMM covariance type is selected internally using BIC/AIC among configured covariance_types.",
        }
    return {}


def separation_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return {"silhouette_score": np.nan, "davies_bouldin_index": np.nan, "calinski_harabasz_score": np.nan}
    return {
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
    }


def balance_metrics(labels: np.ndarray, k: int, min_cluster_size: int) -> Dict[str, Any]:
    vals, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    props = counts / n
    entropy = float(-(props * np.log(props + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0
    min_size = int(counts.min())
    max_size = int(counts.max())
    adequate = min(1.0, min_size / max(min_cluster_size, 1))
    balance = 0.70 * entropy + 0.30 * adequate
    return {
        "cluster_sizes": ";".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "min_cluster_ratio": float(min_size / n),
        "max_cluster_ratio": float(max_size / n),
        "cluster_size_entropy": entropy,
        "small_cluster_count": int((counts < min_cluster_size).sum()),
        "balance_score": float(balance),
        "preferred_size_ok": bool(min_size >= min_cluster_size),
    }



def fit_predict_for_stability(
    algorithm: str,
    k: int,
    X_scaled: np.ndarray,
    config: Dict[str, Any],
    seed: int,
    base_model: Any | None = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Faster refit used only inside repeated subsampling stability.

    Full evaluation still uses the configured model parameters. For GMM, the
    covariance type selected by BIC on the full-data fit is reused during
    subsampling and n_init is reduced to 1. This keeps the 100-run stability
    test practical while preserving the main covariance-selection logic in the
    full evaluation table.
    """
    algorithm = algorithm.lower()
    if algorithm == "kmeans":
        # Stability runs only need relative robustness; n_init=1 keeps 100-run
        # subsampling practical while full evaluation still uses n_init from config.
        km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=300)
        labels = km.fit_predict(X_scaled)
        return km, labels.astype(int), np.full(len(labels), np.nan)
    if algorithm == "ahc":
        params_ahc = model_params(config, "ahc")
        model = make_ahc(k, str(params_ahc.get("linkage", "ward")), str(params_ahc.get("metric", "euclidean")))
        labels = model.fit_predict(X_scaled)
        return model, labels.astype(int), np.full(len(labels), np.nan)

    params = model_params(config, "gmm")
    cov_type = getattr(base_model, "selected_covariance_type_", None) or getattr(base_model, "covariance_type", None) or "diag"
    gm = GaussianMixture(
        n_components=k,
        random_state=seed,
        n_init=1,
        max_iter=int(params.get("max_iter", 500)),
        covariance_type=str(cov_type),
        reg_covar=float(params.get("reg_covar", 1e-6)),
    )
    labels = gm.fit_predict(X_scaled)
    confidence = gm.predict_proba(X_scaled).max(axis=1)
    gm.selected_covariance_type_ = str(cov_type)
    gm.selection_criterion_ = "stability_reuse_full_fit_covariance"
    return gm, labels.astype(int), confidence

def subsampling_stability_ari(
    algorithm: str,
    k: int,
    X_scaled: np.ndarray,
    base_labels: np.ndarray,
    config: Dict[str, Any],
    base_model: Any | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate cluster robustness with repeated subsampling.

    This replaces the weaker seed-only stability check. It is meaningful for
    deterministic algorithms such as AHC because the model is refit on 80%
    subsamples and compared against the labels from the full-data solution.
    ARI is permutation-invariant, so it is safe when cluster IDs are swapped.
    """
    cfg = config.get("clustering", {})
    st_cfg = cfg.get("stability", {}) or {}
    base_seed = int(cfg.get("random_state", 2025))
    n_runs = int(st_cfg.get("n_runs", 50))
    sample_fraction = float(st_cfg.get("sample_fraction", 0.80))
    n = int(X_scaled.shape[0])
    sample_size = max(k + 1, int(round(n * sample_fraction)))
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(base_seed + 7701 + int(k) * 101 + sum(ord(ch) for ch in str(algorithm)))

    scores: List[float] = []
    rows: List[Dict[str, Any]] = []
    for run_idx in range(n_runs):
        subset_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        _, subset_labels, _ = fit_predict_for_stability(algorithm, k, X_scaled[subset_idx], config, base_seed + 1000 + run_idx, base_model=base_model)
        ari = float(adjusted_rand_score(base_labels[subset_idx], subset_labels))
        scores.append(ari)
        rows.append({
            "stability_method": "subsampling_ari",
            "run": run_idx + 1,
            "sample_fraction": sample_fraction,
            "sample_size": sample_size,
            "ari_vs_full_labels": ari,
        })

    stats = {
        "stability_method": "subsampling_ari",
        "stability_n_runs": n_runs,
        "stability_sample_fraction": sample_fraction,
        "stability_sample_size": sample_size,
        "stability_ari_mean": float(np.mean(scores)) if scores else np.nan,
        "stability_ari_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "stability_ari_min": float(np.min(scores)) if scores else np.nan,
        "stability_ari_max": float(np.max(scores)) if scores else np.nan,
    }
    return stats, rows


def scaled_composite(metrics: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    w = config["clustering"].get("composite_weights", {})
    out = metrics.copy()
    out["silhouette_norm"] = minmax(out["silhouette_score"])
    out["davies_bouldin_norm"] = minmax(out["davies_bouldin_index"], invert=True)
    out["calinski_harabasz_norm"] = minmax(out["calinski_harabasz_score"])
    out["balance_norm"] = minmax(out["balance_score"])
    out["stability_norm"] = minmax(out["stability_ari_mean"])
    out["composite_score_final"] = (
        float(w.get("silhouette", 0.30)) * out["silhouette_norm"]
        + float(w.get("davies_bouldin", 0.20)) * out["davies_bouldin_norm"]
        + float(w.get("calinski_harabasz", 0.15)) * out["calinski_harabasz_norm"]
        + float(w.get("balance", 0.20)) * out["balance_norm"]
        + float(w.get("stability", 0.15)) * out["stability_norm"]
    )
    return out



def build_cluster_labels(profile: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Build safe, report-friendly labels for clusters.

    The internal scikit-learn label stays 0,1,2,... but the report/dashboard uses
    Cluster 1, Cluster 2, ... and domain labels. Labels are descriptive and
    relative; they are not rankings and not final installation feasibility.
    """
    p = profile.copy()
    rows = []

    # Special handling for the final k=3 environmental context scenario. Labels are
    # assigned from the actual cluster profile, not manually forcing cluster IDs.
    # For the revised main feature set, thermal_heat_excess/hot_day_35c are no
    # longer used as main clustering features; temperature_p90_c represents heat.
    has_core_context_features = all(c in p.columns for c in [
        "mean_daily_irradiance_kwh_m2", "monthly_irradiance_cv",
        "temperature_p90_c", "rainy_day_ratio",
        "relative_humidity_mean_pct", "elevation_m"
    ])
    if len(p) == 3 and has_core_context_features:
        high_rad_dry_score = (
            p["mean_daily_irradiance_kwh_m2"].rank(ascending=False)
            + p["rainy_day_ratio"].rank(ascending=True)
            + p["relative_humidity_mean_pct"].rank(ascending=True)
        )
        high_rad_cluster = int(p.loc[high_rad_dry_score.idxmin(), "cluster"])
        cool_wet_cluster = int(p.loc[(
            p["elevation_m"].rank(ascending=False)
            + p["rainy_day_ratio"].rank(ascending=False)
            + p["relative_humidity_mean_pct"].rank(ascending=False)
            + p["temperature_p90_c"].rank(ascending=True)
        ).idxmin(), "cluster"])
        hot_lowland_cluster = int(p.loc[(
            p["temperature_p90_c"].rank(ascending=False)
            + p["elevation_m"].rank(ascending=True)
            + p["relative_humidity_mean_pct"].rank(ascending=False)
        ).idxmin(), "cluster"])
        label_map = {
            high_rad_cluster: "Radiasi relatif tinggi dan atmosfer relatif lebih kering",
            cool_wet_cluster: "Sejuk-basah, elevasi relatif tinggi, radiasi relatif lebih rendah",
            hot_lowland_cluster: "Dataran rendah lebih panas-lembap, perlu perhatian termal lanjutan",
        }
    else:
        label_map = {}

    means = {f: profile[f].mean() for f in features if f in profile.columns}
    stds = {f: profile[f].std(ddof=0) if profile[f].std(ddof=0) != 0 else 1.0 for f in features if f in profile.columns}

    for _, row in profile.iterrows():
        cluster = int(row["cluster"])
        cluster_display = int(cluster + 1)
        notes = []

        if cluster in label_map:
            short_label = label_map[cluster]
        else:
            parts = []
            def rel(feature: str) -> float:
                if feature not in row or pd.isna(row[feature]):
                    return 0.0
                return float((row[feature] - means.get(feature, row[feature])) / stds.get(feature, 1.0))
            if rel("mean_daily_irradiance_kwh_m2") > 0.45:
                parts.append("radiasi relatif tinggi")
            elif rel("mean_daily_irradiance_kwh_m2") < -0.45:
                parts.append("radiasi relatif rendah")
            else:
                parts.append("radiasi sedang")
            if rel("thermal_heat_excess_mean_c") > 0.45:
                parts.append("tekanan panas relatif tinggi")
            elif rel("thermal_risk_proxy_mean") > 0.45:
                parts.append("risiko termal relatif tinggi")
            elif rel("temperature_p90_c") > 0.45:
                parts.append("lebih panas")
            elif rel("temperature_p90_c") < -0.45:
                parts.append("lebih sejuk")
            if rel("rainy_day_ratio") > 0.45:
                parts.append("lebih basah")
            elif rel("rainy_day_ratio") < -0.45:
                parts.append("relatif lebih kering")
            if rel("elevation_m") > 0.45:
                parts.append("elevasi relatif tinggi")
            short_label = ", ".join(parts[:4])

        if int(row.get("n_regions", 0)) < 8:
            notes.append("jumlah anggota kecil; interpretasi perlu hati-hati")
        notes.append("label bersifat deskriptif/indikatif, bukan ranking dan bukan keputusan kelayakan final PLTS")

        rows.append({
            "cluster": cluster,
            "cluster_display": cluster_display,
            "cluster_display_name": f"Cluster {cluster_display}",
            "cluster_name_short": short_label,
            "cluster_label": f"Cluster {cluster_display} - {short_label}",
            "interpretation_note": "; ".join(notes),
        })
    return pd.DataFrame(rows)

def evaluate_and_cluster(config: Dict[str, Any], run_name: str | None = None) -> None:
    rd = run_dir(config, run_name)
    out = ensure_dir(rd / "03_clustering")
    pdir = processed_dir(config)
    summary_path = pdir / config["files"]["summary_output"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found. Run step 02 first: {summary_path}")
    summary = pd.read_csv(summary_path)

    cfg = config["clustering"]
    feature_sets = cfg["feature_sets"]
    primary_feature_set = cfg["primary_feature_set"]
    primary_scaler = cfg["primary_scaler"]
    evaluation_feature_sets = cfg.get("evaluation_feature_sets", [primary_feature_set])
    evaluation_scalers = cfg.get("evaluation_scalers", [primary_scaler])
    algorithms = cfg["algorithms"]
    k_values = [int(x) for x in cfg.get("k_values", [2, 3, 4, 5])]
    random_state = int(cfg.get("random_state", 2025))
    min_cluster_size = int(cfg.get("min_cluster_size_preferred", 8))

    rows = []
    impute_rows = []
    stability_detail_rows = []
    transformed_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, np.ndarray, Any, List[str]]] = {}

    for fs_name, spec in feature_sets.items():
        if fs_name not in evaluation_feature_sets:
            continue
        features = list(spec["features"])
        X, imp = prepare_matrix(summary, features)
        imp.insert(0, "feature_set", fs_name)
        impute_rows.append(imp)
        for scaler_name in evaluation_scalers:
            scaler = make_scaler(scaler_name)
            X_scaled = scaler.fit_transform(X)
            transformed_cache[(fs_name, scaler_name)] = (X, X_scaled, scaler, features)
            for alg in algorithms:
                for k in k_values:
                    model, labels, conf = fit_predict(alg, k, X_scaled, config, random_state)
                    row = {
                        "feature_set": fs_name,
                        "feature_set_description": spec.get("description", ""),
                        "scaler": scaler_name,
                        "algorithm": alg,
                        "n_clusters": k,
                        "features_used": ";".join(features),
                        "n_features_used": len(features),
                    }
                    row.update(separation_metrics(X_scaled, labels))
                    row.update(balance_metrics(labels, k, min_cluster_size))
                    row.update(model_info_row(model, alg))
                    st_stats, st_rows = subsampling_stability_ari(alg, k, X_scaled, labels, config, base_model=model)
                    row.update(st_stats)
                    for st in st_rows:
                        st.update({
                            "feature_set": fs_name,
                            "scaler": scaler_name,
                            "algorithm": alg,
                            "n_clusters": k,
                        })
                    stability_detail_rows.extend(st_rows)
                    rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics = scaled_composite(metrics, config)

    # Transparent final selection policy without custom accept_if thresholds.
    # The final domain-selected candidate is reported as a trade-off decision,
    # while metric-best candidates are still saved separately for honesty.
    primary_metrics = metrics[(metrics["feature_set"] == primary_feature_set) & (metrics["scaler"] == primary_scaler)].copy()
    ms_cfg = cfg.get("model_selection", {}) or {}
    domain_cfg = ms_cfg.get("domain_selected", {}) or {}
    dc_alg = str(domain_cfg.get("algorithm", "kmeans"))
    dc_k = int(domain_cfg.get("n_clusters", 3))
    domain_reason = str(domain_cfg.get("reason", "Domain-selected candidate for interpretability."))
    domain_row = primary_metrics[(primary_metrics["algorithm"] == dc_alg) & (primary_metrics["n_clusters"] == dc_k)].copy()

    if not domain_row.empty:
        selected = domain_row.iloc[0].to_dict()
        selection_reason = (
            "Multi-criteria trade-off without custom accept_if thresholds. "
            f"Selected {dc_alg.upper()} k={dc_k} on the primary feature set because: {domain_reason} "
            "Metric-best alternatives remain reported separately; this selection is not claimed as the absolute metric-best when another candidate scores higher."
        )
    else:
        valid = primary_metrics[primary_metrics["preferred_size_ok"] == True].copy()
        if valid.empty:
            valid = primary_metrics.copy()
        selected = valid.sort_values(["composite_score_final", "silhouette_score", "davies_bouldin_index"], ascending=[False, False, True]).iloc[0].to_dict()
        selection_reason = (
            "Configured domain-selected candidate was not found in the evaluated scenario, so the best valid primary-scenario model was selected by composite score."
        )

    fs = str(selected["feature_set"])
    scaler_name = str(selected["scaler"])
    alg = str(selected["algorithm"])
    k = int(selected["n_clusters"])
    X, X_scaled, scaler, features = transformed_cache[(fs, scaler_name)]
    model, labels, confidence = fit_predict(alg, k, X_scaled, config, random_state)

    result = summary.copy()
    result["cluster"] = labels.astype(int)
    result["selected_feature_set"] = fs
    result["selected_scaler"] = scaler_name
    result["selected_algorithm"] = alg
    result["selected_n_clusters"] = k
    result["cluster_confidence"] = confidence

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    result["pca_1"] = coords[:, 0]
    result["pca_2"] = coords[:, 1]
    pca_explained = {"pca_1_explained_ratio": float(pca.explained_variance_ratio_[0]), "pca_2_explained_ratio": float(pca.explained_variance_ratio_[1])}

    profile = result.groupby("cluster")[features].mean().reset_index()
    counts = result.groupby("cluster").size().reset_index(name="n_regions")
    dom = (
        result.groupby(["cluster", "island_group_corrected"]).size().reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False]).drop_duplicates("cluster")
        .rename(columns={"island_group_corrected": "dominant_island_group", "count": "dominant_island_count"})
    )
    profile = counts.merge(profile, on="cluster").merge(dom, on="cluster", how="left")
    labels_df = build_cluster_labels(profile, features)
    profile = profile.merge(labels_df, on="cluster", how="left")
    result = result.merge(labels_df[["cluster", "cluster_display", "cluster_display_name", "cluster_name_short", "cluster_label", "interpretation_note"]], on="cluster", how="left")

    island_dist = result.pivot_table(index="cluster", columns="island_group_corrected", values="region_id", aggfunc="count", fill_value=0).reset_index()
    island_dist = island_dist.merge(labels_df[["cluster", "cluster_display_name"]], on="cluster", how="left")
    first_cols = ["cluster", "cluster_display_name"]
    other_cols = [c for c in island_dist.columns if c not in first_cols]
    island_dist = island_dist[first_cols + other_cols]

    metrics = metrics.sort_values(["feature_set", "scaler", "algorithm", "n_clusters"]).reset_index(drop=True)
    metrics["is_selected_final"] = (
        (metrics["feature_set"] == fs)
        & (metrics["scaler"] == scaler_name)
        & (metrics["algorithm"] == alg)
        & (metrics["n_clusters"] == k)
    )
    metrics["is_domain_candidate"] = (
        (metrics["feature_set"] == primary_feature_set)
        & (metrics["scaler"] == primary_scaler)
        & (metrics["algorithm"] == dc_alg)
        & (metrics["n_clusters"] == dc_k)
    )

    # Metric-best alternatives are saved separately so the report can be honest
    # about the trade-off between metric-best and final selected model.
    primary_ranked = primary_metrics.sort_values(["composite_score_final", "silhouette_score", "davies_bouldin_index"], ascending=[False, False, True]).copy()
    metric_best_primary = primary_ranked.iloc[0].to_dict() if not primary_ranked.empty else {}
    metric_best_overall = metrics.sort_values(["composite_score_final", "silhouette_score", "davies_bouldin_index"], ascending=[False, False, True]).iloc[0].to_dict()

    # Feature sensitivity analysis: compare final k/algorithm under alternative
    # feature sets. This does not change the selected model automatically; it is
    # an audit showing whether the final segmentation is robust when redundant
    # or thermal-interaction features are changed.
    sensitivity_rows = []
    sensitivity_names = [fs] + [name for name in cfg.get("sensitivity_feature_sets", []) if name != fs]
    for sens_fs_name in sensitivity_names:
        if sens_fs_name not in feature_sets:
            continue
        sens_spec = feature_sets[sens_fs_name]
        sens_features = list(sens_spec["features"])
        sens_X, _ = prepare_matrix(summary, sens_features)
        sens_scaler = make_scaler(scaler_name)
        sens_X_scaled = sens_scaler.fit_transform(sens_X)
        _, sens_labels, _ = fit_predict(alg, k, sens_X_scaled, config, random_state)
        sens_row = {
            "feature_set": sens_fs_name,
            "feature_set_description": sens_spec.get("description", ""),
            "scaler": scaler_name,
            "algorithm": alg,
            "n_clusters": k,
            "features_used": ";".join(sens_features),
            "n_features_used": len(sens_features),
            "ari_vs_selected_final_labels": float(adjusted_rand_score(labels, sens_labels)),
            "is_selected_feature_set": bool(sens_fs_name == fs),
        }
        sens_row.update(separation_metrics(sens_X_scaled, sens_labels))
        sens_row.update(balance_metrics(sens_labels, k, min_cluster_size))
        sensitivity_rows.append(sens_row)
    feature_sensitivity = pd.DataFrame(sensitivity_rows).sort_values(["is_selected_feature_set", "ari_vs_selected_final_labels"], ascending=[False, False])

    tradeoff = primary_metrics[(primary_metrics["algorithm"] == dc_alg) & (primary_metrics["n_clusters"].isin([2, dc_k]))].copy()
    tradeoff["role"] = tradeoff["n_clusters"].apply(lambda x: "metric_best_candidate" if int(x) == 2 else "selected_domain_candidate")
    tradeoff = tradeoff.sort_values("n_clusters")

    save_csv(metrics, out / "model_comparison_metrics.csv")
    stability_cols = [c for c in [
        "feature_set", "scaler", "algorithm", "n_clusters",
        "stability_method", "stability_n_runs", "stability_sample_fraction",
        "stability_sample_size", "stability_ari_mean", "stability_ari_std",
        "stability_ari_min", "stability_ari_max"
    ] if c in metrics.columns]
    save_csv(metrics[stability_cols].copy(), out / "stability_subsampling_ari.csv")
    if stability_detail_rows and config.get("outputs", {}).get("keep_stability_detail", False):
        save_csv(pd.DataFrame(stability_detail_rows), out / "stability_subsampling_ari_detail.csv")
    save_csv(primary_ranked, out / "primary_scenario_metrics.csv")
    save_csv(pd.DataFrame([metric_best_primary]), out / "metric_best_primary_model_summary.csv")
    save_csv(pd.DataFrame([metric_best_overall]), out / "metric_best_overall_model_summary.csv")
    save_csv(tradeoff, out / "k2_vs_k3_tradeoff.csv")
    save_csv(feature_sensitivity, out / "feature_sensitivity_comparison.csv")
    save_csv(result, out / "clustering_results.csv")
    save_csv(profile, out / "cluster_profile.csv")
    save_csv(labels_df, out / "cluster_interpretation.csv")
    save_csv(island_dist, out / "cluster_island_group_distribution.csv")
    if impute_rows and config.get("outputs", {}).get("keep_extended_audit", False):
        save_csv(pd.concat(impute_rows, ignore_index=True), out / "feature_imputation_audit.csv")

    best_by_algo = metrics.sort_values("composite_score_final", ascending=False).drop_duplicates(["feature_set", "scaler", "algorithm"])
    save_csv(best_by_algo, out / "best_by_algorithm_within_scenarios.csv")

    save_csv(result, pdir / config["files"].get("clustered_output", "context_aware_clustered_regions.csv"))
    joblib.dump({"model": model, "scaler": scaler, "features": features, "config": config, "pca": pca}, out / "selected_clustering_artifacts.joblib")

    selected_summary = {
        "metric_best_primary_algorithm": metric_best_primary.get("algorithm"),
        "metric_best_primary_n_clusters": int(metric_best_primary.get("n_clusters")) if metric_best_primary else None,
        "metric_best_primary_composite_score_final": float(metric_best_primary.get("composite_score_final")) if metric_best_primary else None,
        "metric_best_primary_silhouette_score": float(metric_best_primary.get("silhouette_score")) if metric_best_primary else None,
        "metric_best_primary_davies_bouldin_index": float(metric_best_primary.get("davies_bouldin_index")) if metric_best_primary else None,
        "metric_best_primary_calinski_harabasz_score": float(metric_best_primary.get("calinski_harabasz_score")) if metric_best_primary else None,
        "metric_best_primary_cluster_sizes": metric_best_primary.get("cluster_sizes"),
        "selected_feature_set": fs,
        "selected_scaler": scaler_name,
        "selected_algorithm": alg,
        "selected_n_clusters": k,
        "selection_reason": selection_reason,
        "silhouette_score": float(selected["silhouette_score"]),
        "davies_bouldin_index": float(selected["davies_bouldin_index"]),
        "calinski_harabasz_score": float(selected["calinski_harabasz_score"]),
        "balance_score": float(selected["balance_score"]),
        "stability_method": selected.get("stability_method"),
        "stability_n_runs": int(selected.get("stability_n_runs", 0)) if not pd.isna(selected.get("stability_n_runs", np.nan)) else None,
        "stability_sample_fraction": float(selected.get("stability_sample_fraction", np.nan)) if not pd.isna(selected.get("stability_sample_fraction", np.nan)) else None,
        "stability_ari_mean": float(selected["stability_ari_mean"]),
        "stability_ari_std": float(selected.get("stability_ari_std", np.nan)),
        "stability_ari_min": float(selected.get("stability_ari_min", np.nan)),
        "stability_ari_max": float(selected.get("stability_ari_max", np.nan)),
        "composite_score_final": float(selected["composite_score_final"]),
        "cluster_sizes": selected["cluster_sizes"],
        "preferred_size_ok": bool(selected["preferred_size_ok"]),
        "features_used": features,
        "pca_explained": pca_explained,
        "note": "Composite score is not accuracy. Clustering result is an indicative segmentation, not final PLTS installation feasibility.",
    }
    save_csv(pd.DataFrame([selected_summary]), out / "selected_model_summary.csv")
    save_csv(pd.DataFrame([selected_summary]), out / "selected_final_model_summary.csv")
    save_csv(labels_df, out / "cluster_label_mapping.csv")
    write_json(selected_summary, out / "selected_model_summary.json")

    cleanup_essential_outputs(config, rd)
    print(
        "STEP 03 selesai. Selected:", alg,
        "k=", k,
        "feature_set=", fs,
        "scaler=", scaler_name,
        "silhouette=", round(float(selected["silhouette_score"]), 4),
        "stability=", round(float(selected["stability_ari_mean"]), 4),
    )


# =============================================================================
# Step 04 - Visualizations
# =============================================================================

def generate_visuals(config: Dict[str, Any], run_name: str | None = None) -> None:
    import matplotlib.pyplot as plt

    rd = run_dir(config, run_name)
    cdir = rd / "03_clustering"
    audit_dir = rd / "02_summary_audit"
    fdir = ensure_dir(rd / "04_figures")
    mdir = ensure_dir(rd / "05_maps")
    result = pd.read_csv(cdir / "clustering_results.csv")
    metrics = pd.read_csv(cdir / "model_comparison_metrics.csv")
    selected = read_json(cdir / "selected_model_summary.json")
    profile = pd.read_csv(cdir / "cluster_profile.csv")
    features = selected.get("features_used", config["feature_engineering"]["balanced_context_features"])
    primary = metrics[(metrics["feature_set"] == selected["selected_feature_set"]) & (metrics["scaler"] == selected["selected_scaler"])]

    # Metric line charts for the selected feature set/scaler.
    metric_specs = [
        ("silhouette_score", "Silhouette Score", "comparison_silhouette_score.png"),
        ("davies_bouldin_index", "Davies-Bouldin Index", "comparison_davies_bouldin_index.png"),
        ("calinski_harabasz_score", "Calinski-Harabasz Score", "comparison_calinski_harabasz_score.png"),
        ("balance_score", "Balance Score", "comparison_balance_score.png"),
        ("stability_ari_mean", "Stability ARI", "comparison_stability_ari.png"),
        ("composite_score_final", "Composite Score Final", "comparison_composite_score_final.png"),
    ]
    for col, ylabel, fname in metric_specs:
        plt.figure(figsize=(9, 5))
        for alg in sorted(primary["algorithm"].unique()):
            sub = primary[primary["algorithm"] == alg].sort_values("n_clusters")
            plt.plot(sub["n_clusters"], sub[col], marker="o", label=alg.upper())
        plt.xlabel("Jumlah Cluster (k)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Algoritma ({selected['selected_feature_set']}, {selected['selected_scaler']})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fdir / fname, dpi=180)
        plt.close()

    # Prepare selected feature matrix for PCA/profile visualizations.
    summary = pd.read_csv(processed_dir(config) / config["files"]["summary_output"])
    X, _ = prepare_matrix(summary, features)
    scaler = make_scaler(selected["selected_scaler"])
    X_scaled = scaler.fit_transform(X)

    # Optional extra figures are disabled in the final thesis package to keep
    # output folders concise. Set outputs.generate_extra_figures=true if needed.
    if config.get("outputs", {}).get("generate_extra_figures", False):
        inertias = []
        for k in config["clustering"].get("k_values", [2, 3, 4, 5]):
            km_params = model_params(config, "kmeans")
            km = KMeans(n_clusters=int(k), random_state=int(km_params.get("random_state", config["clustering"].get("random_state", 2025))), n_init=int(km_params.get("n_init", 50)), max_iter=int(km_params.get("max_iter", 500)))
            km.fit(X_scaled)
            inertias.append({"n_clusters": int(k), "inertia": float(km.inertia_)})
        inertia_df = pd.DataFrame(inertias)
        save_csv(inertia_df, cdir / "kmeans_elbow_inertia.csv")
        plt.figure(figsize=(8, 5))
        plt.plot(inertia_df["n_clusters"], inertia_df["inertia"], marker="o")
        plt.xlabel("Jumlah Cluster (k)")
        plt.ylabel("WCSS / Inertia")
        plt.title("Elbow Method - K-Means")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fdir / "elbow_kmeans_inertia.png", dpi=180)
        plt.close()

        plt.figure(figsize=(11, 6))
        Z = linkage(X_scaled, method="ward")
        dendrogram(Z, no_labels=True, color_threshold=None)
        plt.title("Dendrogram AHC (Ward Linkage) - Fitur Final")
        plt.xlabel("Wilayah")
        plt.ylabel("Jarak Ward")
        plt.tight_layout()
        plt.savefig(fdir / "ahc_dendrogram_ward.png", dpi=180)
        plt.close()

    # PCA scatter.
    plt.figure(figsize=(8, 6))
    for cluster in sorted(result["cluster"].unique()):
        sub = result[result["cluster"] == cluster]
        plt.scatter(sub["pca_1"], sub["pca_2"], s=45, alpha=0.85, label=f"Cluster {int(cluster) + 1}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Sebaran Cluster dalam Ruang PCA")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "pca_cluster_scatter.png", dpi=180)
    plt.close()

    if config.get("outputs", {}).get("generate_extra_figures", False):
        primary_feature_set_name = config.get("clustering", {}).get("primary_feature_set", "core_context_6")
        corr_path = audit_dir / f"{primary_feature_set_name}_correlation_matrix.csv"
        if corr_path.exists():
            corr_df = pd.read_csv(corr_path)
            labels = corr_df.iloc[:, 0].tolist()
            mat = corr_df.drop(columns=corr_df.columns[0]).to_numpy(dtype=float)
            plt.figure(figsize=(9, 7))
            plt.imshow(mat, vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(label="Korelasi")
            plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
            plt.yticks(range(len(labels)), labels, fontsize=8)
            plt.title(f"Heatmap Korelasi Fitur {primary_feature_set_name}")
            plt.tight_layout()
            plt.savefig(fdir / "feature_correlation_matrix.png", dpi=180)
            plt.close()

    # Cluster profile heatmap z-normalized.
    prof = profile.set_index("cluster")[features].copy()
    zprof = (prof - prof.mean()) / prof.std(ddof=0).replace(0, np.nan)
    zprof = zprof.fillna(0.0)
    save_csv(zprof.reset_index(), cdir / "cluster_profile_z_normalized.csv")
    plt.figure(figsize=(10, 6))
    plt.imshow(zprof.to_numpy(), aspect="auto")
    plt.colorbar(label="Z-normalized mean")
    plt.xticks(range(len(features)), features, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(zprof.index)), [f"Cluster {int(i) + 1}" for i in zprof.index])
    plt.title("Heatmap Profil Cluster (Z-normalized)")
    plt.tight_layout()
    plt.savefig(fdir / "cluster_profile_heatmap_z.png", dpi=180)
    plt.close()

    # Island group distribution per cluster.
    island_dist = pd.read_csv(cdir / "cluster_island_group_distribution.csv")
    idf = island_dist.set_index("cluster")
    display_labels = idf["cluster_display_name"].tolist() if "cluster_display_name" in idf.columns else [f"Cluster {int(i)+1}" for i in idf.index]
    idf_numeric = idf.drop(columns=["cluster_display_name"], errors="ignore")
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(idf_numeric))
    for col in idf_numeric.columns:
        vals = idf_numeric[col].to_numpy(dtype=float)
        plt.bar(display_labels, vals, bottom=bottom, label=col)
        bottom += vals
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Wilayah")
    plt.title("Distribusi Island Group per Cluster")
    plt.legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fdir / "cluster_island_group_distribution.png", dpi=180)
    plt.close()

    # Radar chart of normalized profiles.
    labels = features
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    normalized = prof.copy()
    for col in normalized.columns:
        rng = normalized[col].max() - normalized[col].min()
        normalized[col] = 0.5 if np.isclose(rng, 0) else (normalized[col] - normalized[col].min()) / rng
    for cluster, row in normalized.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, label=f"Cluster {int(cluster) + 1}")
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Radar Profil Cluster (Min-Max per Fitur)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    plt.savefig(fdir / "cluster_profile_radar.png", dpi=180)
    plt.close()

    if config.get("outputs", {}).get("generate_extra_figures", False):
        top = result.sort_values("indicative_context_score", ascending=False).head(20)
        plt.figure(figsize=(10, 7))
        plt.barh(top["region_name"].iloc[::-1], top["indicative_context_score"].iloc[::-1])
        plt.xlabel("Indicative Context Score")
        plt.title("Top 20 Wilayah berdasarkan Skor Indikatif")
        plt.tight_layout()
        plt.savefig(fdir / "top_20_indicative_context_score.png", dpi=180)
        plt.close()

    if config.get("outputs", {}).get("generate_maps", False):
        try:
            import folium
            center = [result["centroid_latitude"].mean(), result["centroid_longitude"].mean()]
            fmap = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
            for _, r in result.iterrows():
                popup = (
                    f"<b>{r['region_name']}</b><br>"
                    f"Cluster: {r.get('cluster_display_name', 'Cluster ' + str(int(r['cluster']) + 1))}<br>"
                    f"Label: {r.get('cluster_label','')}<br>"
                    f"Island group: {r['island_group_corrected']}<br>"
                    f"Irradiance: {r['mean_daily_irradiance_kwh_m2']:.2f} kWh/m²/day<br>"
                    f"Rainy day ratio: {r['rainy_day_ratio']:.2f}"
                )
                folium.CircleMarker(
                    location=[r["centroid_latitude"], r["centroid_longitude"]],
                    radius=5,
                    popup=popup,
                    tooltip=f"{r['region_name']} | C{int(r['cluster']) + 1}",
                    fill=True,
                    fill_opacity=0.8,
                ).add_to(fmap)
            fmap.save(mdir / "interactive_cluster_map.html")
        except Exception as exc:
            with open(mdir / "interactive_cluster_map_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(str(exc))

    cleanup_essential_outputs(config, rd)
    print("STEP 04 selesai. Essential figures saved to:", fdir)


# =============================================================================
# Step 05 - Report
# =============================================================================

def generate_report(config: Dict[str, Any], run_name: str | None = None) -> None:
    rd = run_dir(config, run_name)
    report_dir = ensure_dir(rd / "06_reports")
    cdir = rd / "03_clustering"
    raw = read_json(rd / "00_raw_audit" / "raw_audit_summary.json")
    pre = read_json(rd / "01_preprocessing_audit" / "preprocessing_audit_summary.json")
    summ = read_json(rd / "02_summary_audit" / "summary_audit.json")
    selected = read_json(cdir / "selected_model_summary.json")
    profile = pd.read_csv(cdir / "cluster_profile.csv")
    metrics = pd.read_csv(cdir / "model_comparison_metrics.csv")
    island_dist = pd.read_csv(cdir / "cluster_island_group_distribution.csv")
    metric_best_primary_path = cdir / "metric_best_primary_model_summary.csv"
    tradeoff_path = cdir / "k2_vs_k3_tradeoff.csv"
    retention_path = rd / "02_summary_audit" / "feature_retention_decision.csv"
    sensitivity_path = cdir / "feature_sensitivity_comparison.csv"
    stability_path = cdir / "stability_subsampling_ari.csv"
    metric_best_primary = pd.read_csv(metric_best_primary_path).iloc[0].to_dict() if metric_best_primary_path.exists() else {}
    tradeoff = pd.read_csv(tradeoff_path) if tradeoff_path.exists() else pd.DataFrame()
    retention = pd.read_csv(retention_path) if retention_path.exists() else pd.DataFrame()
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    stability_summary = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()

    lines = []
    lines.append("# Final Clustering Report - PLTS Context-Aware")
    lines.append("")
    lines.append("## 1. Ringkasan Dataset")
    lines.append(f"- Jumlah baris raw daily: {raw.get('rows')}")
    lines.append(f"- Jumlah wilayah representatif: {raw.get('unique_regions')}")
    lines.append(f"- Jumlah tanggal: {raw.get('unique_dates')}")
    lines.append(f"- Periode: {raw.get('date_min')} sampai {raw.get('date_max')}")
    lines.append(f"- Missing raw: {raw.get('missing_total')}")
    lines.append(f"- Duplikasi region-date: {raw.get('duplicate_region_date')}")
    lines.append("")
    lines.append("## 2. Preprocessing")
    lines.append(f"- Output clean daily: `{pre.get('output_clean_daily')}`")
    lines.append(f"- Koreksi island_group pada wilayah: {pre.get('island_group_changed_regions')}")
    lines.append(f"- Presipitasi negatif kecil sebelum clip: {pre.get('precipitation_negative_before_clip')}")
    lines.append(f"- Selisih maksimum audit konversi J/m² ke kWh/m²: {pre.get('max_conversion_diff_kwh_m2')}")
    lines.append("")
    lines.append("## 3. Summary Wilayah")
    lines.append(f"- Output summary: `{summ.get('output_summary')}`")
    lines.append(f"- Baris summary: {summ.get('rows')}")
    lines.append(f"- Hari minimum/maksimum per wilayah: {summ.get('min_n_days')} / {summ.get('max_n_days')}")
    lines.append(f"- Feature set audit: {summ.get('feature_audit_set')}")
    lines.append("- Feature set utama final ditentukan oleh config `clustering.primary_feature_set`; pada revisi ini menggunakan `core_context_6` agar fitur utama tidak bergantung pada parameter threshold thermal heat 30 C atau hot day 35 C.")
    lines.append("")
    lines.append("## 4. Model Clustering: Metric-best vs Model Final")
    if metric_best_primary:
        lines.append("### Metric-best pada skenario utama")
        lines.append(f"- Algoritma: {str(metric_best_primary.get('algorithm')).upper()}")
        lines.append(f"- Jumlah cluster: {metric_best_primary.get('n_clusters')}")
        lines.append(f"- Silhouette Score: {metric_best_primary.get('silhouette_score')}")
        lines.append(f"- Davies-Bouldin Index: {metric_best_primary.get('davies_bouldin_index')}")
        lines.append(f"- Calinski-Harabasz Score: {metric_best_primary.get('calinski_harabasz_score')}")
        lines.append(f"- Composite Score Final: {metric_best_primary.get('composite_score_final')}")
        lines.append(f"- Cluster sizes: {metric_best_primary.get('cluster_sizes')}")
        lines.append("")
    lines.append("### Model final terpilih")
    lines.append(f"- Feature set: {selected.get('selected_feature_set')}")
    lines.append(f"- Scaler: {selected.get('selected_scaler')}")
    lines.append(f"- Algoritma: {str(selected.get('selected_algorithm')).upper()}")
    lines.append(f"- Jumlah cluster: {selected.get('selected_n_clusters')}")
    lines.append(f"- Silhouette Score: {selected.get('silhouette_score')}")
    lines.append(f"- Davies-Bouldin Index: {selected.get('davies_bouldin_index')}")
    lines.append(f"- Calinski-Harabasz Score: {selected.get('calinski_harabasz_score')}")
    lines.append(f"- Balance Score: {selected.get('balance_score')}")
    lines.append(f"- Stability ARI Mean: {selected.get('stability_ari_mean')}")
    lines.append(f"- Stability ARI Std: {selected.get('stability_ari_std')}")
    lines.append(f"- Stability method: {selected.get('stability_method')}")
    lines.append(f"- Cluster sizes: {selected.get('cluster_sizes')}")
    lines.append("")
    lines.append("### Catatan Pemilihan Model")
    lines.append(str(selected.get("selection_reason")))
    lines.append("")
    lines.append("> Model final tidak diklaim sebagai model terbaik secara metrik murni apabila terdapat kandidat dengan skor numerik lebih tinggi. Pemilihan dilakukan dengan pendekatan multi-criteria trade-off tanpa threshold accept_if buatan, sehingga metric-best tetap dilaporkan secara transparan.")
    lines.append("")
    if not tradeoff.empty:
        lines.append("### Trade-off k=2 dan k=3 pada skenario utama")
        lines.append(tradeoff[[c for c in ["role", "algorithm", "n_clusters", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_score", "balance_score", "stability_ari_mean", "composite_score_final", "cluster_sizes"] if c in tradeoff.columns]].to_markdown(index=False))
        lines.append("")
    lines.append("> Composite score bukan akurasi. Score ini hanya skor relatif internal untuk membandingkan kandidat clustering. Hasil clustering adalah segmentasi indikatif, bukan keputusan final kelayakan instalasi PLTS.")
    lines.append("")
    if not retention.empty:
        lines.append("## 4B. Catatan Retensi Fitur Berkorelasi Tinggi")
        lines.append("Korelasi antarfitur digunakan sebagai audit redundansi. Fitur tidak dihapus otomatis apabila tetap memiliki makna fisik berbeda dalam konteks PLTS.")
        lines.append(retention.to_markdown(index=False))
        lines.append("")
    if not sensitivity.empty:
        lines.append("## 4C. Feature Sensitivity Analysis")
        lines.append("Analisis ini menguji apakah hasil final terlalu tergantung pada fitur yang berpotensi redundant atau pada thermal risk proxy. Nilai ARI dibandingkan dengan label final terpilih.")
        cols = [c for c in ["feature_set", "n_features_used", "ari_vs_selected_final_labels", "silhouette_score", "davies_bouldin_index", "balance_score", "cluster_sizes"] if c in sensitivity.columns]
        lines.append(sensitivity[cols].to_markdown(index=False))
        lines.append("")
    if not stability_summary.empty:
        lines.append("## 4D. Stability ARI Berbasis Subsampling")
        lines.append("Stabilitas dihitung dengan mengambil subset wilayah berulang, menjalankan clustering ulang, lalu membandingkannya dengan label full-data menggunakan Adjusted Rand Index. Metode ini juga berlaku untuk AHC yang deterministik.")
        sel_stab = stability_summary[(stability_summary["feature_set"] == selected.get("selected_feature_set")) & (stability_summary["scaler"] == selected.get("selected_scaler"))]
        sel_stab = sel_stab.sort_values(["algorithm", "n_clusters"])
        cols = [c for c in ["algorithm", "n_clusters", "stability_ari_mean", "stability_ari_std", "stability_ari_min", "stability_ari_max"] if c in sel_stab.columns]
        lines.append(sel_stab[cols].to_markdown(index=False))
        lines.append("")
    lines.append("## 5. Profil Cluster")
    for _, row in profile.iterrows():
        lines.append(f"### {row.get('cluster_display_name', 'Cluster ' + str(int(row['cluster']) + 1))}")
        lines.append(f"- Jumlah wilayah: {int(row['n_regions'])}")
        lines.append(f"- Label deskriptif: {row.get('cluster_label','')}")
        lines.append(f"- Dominan island group: {row.get('dominant_island_group','')}")
        lines.append(f"- Catatan: {row.get('interpretation_note','')}")
        for f in selected.get("features_used", []):
            if f in row:
                lines.append(f"- {f}: {row[f]:.4f}")
        lines.append("")
    lines.append("## 6. Distribusi Island Group per Cluster")
    lines.append("Catatan: distribusi ini menunjukkan dominasi relatif, bukan berarti cluster ditentukan oleh pulau/kawasan. Clustering tetap berdasarkan fitur lingkungan.")
    lines.append(island_dist.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. Batasan Interpretasi")
    lines.append("- Penelitian ini menggunakan 137 wilayah representatif, bukan seluruh kabupaten/kota Indonesia.")
    lines.append("- Skor dan cluster bersifat potensi awal/indikatif berbasis data lingkungan.")
    lines.append("- Hasil tidak mencakup luas atap, shading, orientasi panel, kemiringan, detail inverter, biaya investasi, tarif listrik, atau payback period.")
    lines.append("- Cluster dengan nilai lebih rendah tidak berarti wilayah tidak dapat menggunakan PLTS; hanya menunjukkan perlunya pertimbangan teknis lanjutan yang lebih hati-hati.")
    lines.append("")
    lines.append("## 8. Output Penting")
    lines.append("- `data/preprocessed/combined_daily_2023_2025_clean.csv`")
    lines.append("- `data/processed/context_aware_summary_2023_2025.csv`")
    lines.append("- `data/processed/context_aware_clustered_regions.csv`")
    lines.append("- `outputs/runs/<run_name>/03_clustering/model_comparison_metrics.csv`")
    lines.append("- `outputs/runs/<run_name>/04_figures/`")
    lines.append("- `outputs/runs/<run_name>/06_reports/final_clustering_report.md`")

    report_path = report_dir / "final_clustering_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    readme_path = report_dir / "README_OUTPUT.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("# Output Folder Guide\n\n")
        f.write("Folder ini berisi hasil run pipeline. Lihat `final_clustering_report.md` untuk ringkasan.\n")

    cleanup_essential_outputs(config, rd)
    print("STEP 05 selesai. Report saved to:", report_path)
