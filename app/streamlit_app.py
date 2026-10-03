from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# Konfigurasi Halaman
# ============================================================

st.set_page_config(
    page_title="Peta Segmentasi Potensi Awal PLTS",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


# ============================================================
# Helper Function
# ============================================================

def list_run_names() -> list[str]:
    """Mengambil daftar folder run pada outputs/runs."""
    if not RUNS_DIR.exists():
        return []
    return sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])


def find_cluster_file(run_name: str) -> Path | None:
    """Mencari file hasil clustering utama."""
    run_dir = RUNS_DIR / run_name
    candidates = [
        run_dir / "03_clustering" / "clustering_results.csv",
        run_dir / "03_clustering" / "clustered_regions.csv",
        run_dir / "03_clustering" / "final_cluster_assignments.csv",
        PROJECT_ROOT / "data" / "processed" / "context_aware_clustered_regions.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    if run_dir.exists():
        for path in run_dir.rglob("*.csv"):
            if "cluster" in path.name.lower():
                return path
    return None


def find_cluster_profile_file(run_name: str) -> Path | None:
    path = RUNS_DIR / run_name / "03_clustering" / "cluster_profile.csv"
    return path if path.exists() else None


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Memilih nama kolom berdasarkan beberapa kemungkinan nama."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def prepare_cluster_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan dataframe agar siap ditampilkan di peta."""
    lat_col = pick_column(df, ["centroid_latitude", "latitude", "lat", "y"])
    lon_col = pick_column(df, ["centroid_longitude", "longitude", "lon", "lng", "x"])
    cluster_col = pick_column(
        df,
        ["cluster", "final_cluster", "cluster_label", "selected_cluster", "kmeans_cluster"],
    )
    region_col = pick_column(
        df,
        ["region_name", "region", "name", "WADMKK", "kabupaten_kota", "kabupaten", "city"],
    )

    missing = []
    if lat_col is None:
        missing.append("latitude / centroid_latitude")
    if lon_col is None:
        missing.append("longitude / centroid_longitude")
    if cluster_col is None:
        missing.append("cluster / final_cluster")
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + ". Periksa file hasil clustering."
        )

    out = df.copy()
    out["latitude"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["cluster_raw"] = pd.to_numeric(out[cluster_col], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude", "cluster_raw"]).copy()

    # Jika label cluster dari program 0,1,2 maka ditampilkan menjadi 1,2,3.
    min_cluster = int(out["cluster_raw"].min())
    if min_cluster == 0:
        out["cluster_display"] = out["cluster_raw"].astype(int) + 1
    else:
        out["cluster_display"] = out["cluster_raw"].astype(int)

    if region_col is None:
        out["wilayah"] = "Wilayah " + out.index.astype(str)
    else:
        out["wilayah"] = out[region_col].astype(str)

    # Warna visual:
    # Cluster 1 = merah, Cluster 2 = biru, Cluster 3 = hijau.
    color_map = {
        1: [220, 53, 69, 190],     # merah
        2: [0, 123, 255, 190],     # biru
        3: [40, 167, 69, 190],     # hijau
    }
    out["color"] = out["cluster_display"].map(color_map)
    out["color"] = out["color"].apply(
        lambda x: x if isinstance(x, list) else [108, 117, 125, 180]
    )
    return out


def load_cluster_profile(run_name: str) -> pd.DataFrame | None:
    profile_path = find_cluster_profile_file(run_name)
    if profile_path is None:
        return None
    try:
        profile = pd.read_csv(profile_path)
        if "cluster_display" not in profile.columns:
            if "cluster" in profile.columns:
                cluster_num = pd.to_numeric(profile["cluster"], errors="coerce")
                if int(cluster_num.min()) == 0:
                    profile["cluster_display"] = cluster_num.astype(int) + 1
                else:
                    profile["cluster_display"] = cluster_num.astype(int)
        return profile
    except Exception:
        return None


def safe_float(row: pd.Series, col: str) -> float | None:
    if col not in row or pd.isna(row[col]):
        return None
    try:
        return float(row[col])
    except Exception:
        return None


def format_profile_values(row: pd.Series) -> list[str]:
    """Membuat ringkasan nilai profil cluster jika cluster_profile.csv tersedia."""
    items = []
    mapping = [
        ("mean_daily_irradiance_kwh_m2", "Radiasi rata-rata", "{:.2f} kWh/m²/hari"),
        ("monthly_irradiance_cv", "Variasi radiasi bulanan", "{:.3f}"),
        ("temperature_p90_c", "Temperature P90", "{:.2f} °C"),
        ("rainy_day_ratio", "Rasio hari hujan", "{:.3f}"),
        ("relative_humidity_mean_pct", "Kelembapan relatif", "{:.2f}%"),
        ("elevation_m", "Elevasi", "{:.1f} m"),
    ]
    for col, label, fmt in mapping:
        val = safe_float(row, col)
        if val is not None:
            items.append(f"{label}: {fmt.format(val)}")
    return items


def fallback_cluster_description(cluster_display: int) -> dict[str, object]:
    """Deskripsi aman jika cluster_profile.csv belum tersedia."""
    if cluster_display == 1:
        return {
            "title": "Cluster 1 (Merah)",
            "border": "#dc3545",
            "bg": "#fff5f5",
            "heading": "#dc3545",
            "bullets": [
                "Radiasi rata-rata relatif lebih tinggi.",
                "Rasio hari hujan relatif lebih rendah.",
                "Kelembapan relatif lebih rendah dibanding cluster lain.",
            ],
            "note": (
                "Cluster ini dapat dibaca sebagai wilayah yang relatif lebih mendukung "
                "potensi awal PLTS dari sisi ketersediaan radiasi dan kondisi atmosfer "
                "yang lebih kering. Namun, hasil ini tetap bersifat indikatif."
            ),
        }
    if cluster_display == 2:
        return {
            "title": "Cluster 2 (Biru)",
            "border": "#007bff",
            "bg": "#f3f8ff",
            "heading": "#007bff",
            "bullets": [
                "Radiasi cenderung sedang atau lebih rendah dibanding cluster radiasi tinggi.",
                "Rasio hari hujan dan kelembapan relatif cenderung lebih tinggi.",
                "Beberapa wilayah dapat merepresentasikan kondisi lebih basah, sejuk, atau berelevasi lebih tinggi.",
            ],
            "note": (
                "Cluster ini tidak dimaknai sebagai wilayah buruk untuk PLTS, tetapi "
                "sebagai kelompok wilayah dengan karakter lingkungan yang perlu dianalisis "
                "bersama faktor hujan, kelembapan, suhu, dan elevasi."
            ),
        }
    return {
        "title": "Cluster 3 (Hijau)",
        "border": "#28a745",
        "bg": "#f4fff7",
        "heading": "#28a745",
        "bullets": [
            "Radiasi masih tersedia dan tetap relevan untuk potensi awal PLTS.",
            "Kondisi lingkungan dapat menunjukkan suhu atau kelembapan yang lebih tinggi.",
            "Beberapa wilayah memerlukan perhatian lanjutan pada aspek cuaca dan karakter lingkungan.",
        ],
        "note": (
            "Cluster ini dapat dibaca sebagai wilayah yang tetap memiliki potensi radiasi, "
            "tetapi memiliki karakter lingkungan yang perlu diperhatikan lebih lanjut "
            "dalam analisis teknis lanjutan."
        ),
    }


def render_cluster_card(cluster_display: int, profile: pd.DataFrame | None) -> None:
    desc = fallback_cluster_description(cluster_display)
    bullets = list(desc["bullets"])
    title_extra = ""

    if profile is not None and "cluster_display" in profile.columns:
        row_match = profile[profile["cluster_display"].astype(int) == int(cluster_display)]
        if not row_match.empty:
            row = row_match.iloc[0]
            if "cluster_name_short" in row and pd.notna(row["cluster_name_short"]):
                title_extra = f"<p style='margin:0 0 8px 0;'><b>{row['cluster_name_short']}</b></p>"
            values = format_profile_values(row)
            if values:
                bullets = values

    bullet_html = "".join(f"<li>{b}</li>" for b in bullets)
    html = f"""
    <div style="border:1px solid {desc['border']}; border-radius:10px; padding:14px; background-color:{desc['bg']}; min-height:330px;">
        <h4 style="color:{desc['heading']}; margin-top:0;">{desc['title']}</h4>
        {title_extra}
        <p><b>Karakter utama:</b></p>
        <ul>{bullet_html}</ul>
        <p>{desc['note']}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Load Data
# ============================================================

run_names = list_run_names()

if not run_names:
    st.error("Folder outputs/runs belum ditemukan atau belum ada hasil pipeline.")
    st.stop()

default_index = 0
if "final_run_core_context_6" in run_names:
    default_index = run_names.index("final_run_core_context_6")

run_name = st.sidebar.selectbox("Pilih Run", run_names, index=default_index)
cluster_file = find_cluster_file(run_name)

if cluster_file is None:
    st.error(
        f"Tidak menemukan file hasil clustering untuk run: {run_name}. "
        "Pastikan script 03_run_clustering.py sudah dijalankan."
    )
    st.stop()

try:
    raw_df = pd.read_csv(cluster_file)
    df = prepare_cluster_dataframe(raw_df)
    profile_df = load_cluster_profile(run_name)
except Exception as exc:
    st.error(f"Gagal membaca data cluster: {exc}")
    st.stop()

if df.empty:
    st.warning("Data cluster kosong atau koordinat tidak valid.")
    st.stop()


# ============================================================
# Tampilan Utama
# ============================================================

st.title("Peta Segmentasi Potensi Awal PLTS")
st.caption(
    "Dashboard ini menampilkan peta hasil clustering wilayah berdasarkan karakteristik lingkungan. "
    "Warna cluster digunakan sebagai pembeda visual, bukan peringkat kelayakan final PLTS."
)

st.markdown(
    """
    <div style="line-height:1.8; font-size:16px; margin-bottom:10px;">
    <span style="color:rgb(220,53,69); font-weight:bold;">●</span> Cluster 1 &nbsp;&nbsp;
    <span style="color:rgb(0,123,255); font-weight:bold;">●</span> Cluster 2 &nbsp;&nbsp;
    <span style="color:rgb(40,167,69); font-weight:bold;">●</span> Cluster 3
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Peta Cluster
# ============================================================

tooltip = {
    "html": """
    <b>{wilayah}</b><br/>
    Cluster: {cluster_display}<br/>
    Latitude: {latitude}<br/>
    Longitude: {longitude}
    """,
    "style": {"backgroundColor": "white", "color": "black"},
}

mid_lat = float(df["latitude"].mean())
mid_lon = float(df["longitude"].mean())

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[longitude, latitude]",
    get_fill_color="color",
    get_radius=45000,
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=4, pitch=0)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="light",
)

st.pydeck_chart(deck, use_container_width=True)


# ============================================================
# Keterangan Karakteristik Cluster
# ============================================================

st.subheader("Keterangan Karakteristik Cluster")
st.caption(
    "Keterangan ini membantu membaca makna visual cluster berdasarkan profil fitur lingkungan. "
    "Cluster tidak menunjukkan peringkat kelayakan final PLTS, tetapi hanya segmentasi karakteristik wilayah."
)

col1, col2, col3 = st.columns(3)
with col1:
    render_cluster_card(1, profile_df)
with col2:
    render_cluster_card(2, profile_df)
with col3:
    render_cluster_card(3, profile_df)


# ============================================================
# Ringkasan dan Tabel Wilayah
# ============================================================

cluster_counts = (
    df["cluster_display"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Jumlah Wilayah")
)

with st.expander("Lihat jumlah wilayah per cluster"):
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

with st.expander("Lihat daftar wilayah dan cluster"):
    table_df = df[["wilayah", "cluster_display", "latitude", "longitude"]].copy()
    table_df = table_df.rename(
        columns={
            "wilayah": "Wilayah",
            "cluster_display": "Cluster",
            "latitude": "Latitude",
            "longitude": "Longitude",
        }
    )
    st.dataframe(
        table_df.sort_values(["Cluster", "Wilayah"]),
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "Catatan: hasil cluster menunjukkan segmentasi karakteristik lingkungan wilayah. "
    "Dashboard ini tidak menyatakan prediksi produksi kWh aktual dan tidak menjadi "
    "rekomendasi final pemasangan PLTS."
)
