import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Proyek Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --app-primary: var(--primary-color);
        --app-text: var(--text-color);
        --app-bg: var(--background-color);
        --app-surface: var(--secondary-background-color);
        --app-border: rgba(59, 130, 246, 0.35);
    }
    .main-header {
        font-size: 2.2rem;
        color: var(--app-primary);
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.35rem;
        color: var(--app-primary);
        margin-top: 1.2rem;
    }
    .info-box {
        background: linear-gradient(
            180deg,
            color-mix(in srgb, var(--app-surface) 85%, var(--app-bg) 15%) 0%,
            var(--app-surface) 100%
        );
        color: var(--app-text);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--app-border);
        border-left: 5px solid var(--app-primary);
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
        color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.22);
        text-align: center;
    }
    .metric-card h3,
    .metric-card p {
        color: #ffffff;
        margin: 0.2rem 0;
    }
    .stage-card {
        background: var(--app-surface);
        color: var(--app-text);
        padding: 0.55rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-left: 4px solid #22c55e;
    }
    .creator-card {
        background: linear-gradient(
            145deg,
            color-mix(in srgb, var(--app-surface) 82%, var(--app-bg) 18%) 0%,
            var(--app-surface) 100%
        );
        border: 1px solid var(--app-border);
        border-left: 5px solid #4f46e5;
        border-radius: 12px;
        padding: 0.9rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12);
        margin-top: 0.3rem;
    }
    .creator-title {
        font-size: 0.78rem;
        color: color-mix(in srgb, var(--app-text) 75%, transparent);
        letter-spacing: 0.4px;
        margin-bottom: 0.35rem;
    }
    .creator-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--app-text);
        margin-bottom: 0.15rem;
    }
    .creator-role {
        font-size: 0.78rem;
        color: color-mix(in srgb, var(--app-text) 78%, transparent);
        margin-bottom: 0.45rem;
    }
    .creator-link a {
        font-size: 0.86rem;
        color: var(--app-primary) !important;
        text-decoration: none;
        font-weight: 600;
    }
    .creator-link a:hover {
        text-decoration: underline;
    }
    .empty-state-card {
        text-align: center;
        padding: 3rem;
        border-radius: 12px;
        background: linear-gradient(
            180deg,
            color-mix(in srgb, var(--app-surface) 88%, var(--app-bg) 12%) 0%,
            var(--app-surface) 100%
        );
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: var(--app-text);
        margin-bottom: 1rem;
    }
    .muted-text {
        color: color-mix(in srgb, var(--app-text) 78%, transparent);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ProjectStage:
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params["optimistic"]
        self.most_likely = base_params["most_likely"]
        self.pessimistic = base_params["pessimistic"]
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations,
        )

        for risk_params in self.risk_factors.values():
            if risk_params["type"] == "discrete":
                probability = risk_params["probability"]
                impact = risk_params["impact"]
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration,
                )

            elif risk_params["type"] == "continuous":
                mean = risk_params["mean"]
                std = risk_params["std"]
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)

        return base_duration * risk_multiplier


class MonteCarloProjectSimulation:
    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self.initialize_stages()

    def initialize_stages(self):
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config["base_params"],
                risk_factors=config.get("risk_factors", {}),
                dependencies=config.get("dependencies", []),
            )

    def run_simulation(self):
        results = pd.DataFrame(index=range(self.num_simulations))

        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)

        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))

        # Topological sort sederhana berdasarkan dependencies.
        processed_stages = set()
        stage_order = []

        while len(processed_stages) < len(self.stages):
            progressed = False
            for stage_name in self.stages.keys():
                if stage_name in processed_stages:
                    continue

                deps = self.stages[stage_name].dependencies
                if all(dep in processed_stages for dep in deps):
                    stage_order.append(stage_name)
                    processed_stages.add(stage_name)
                    progressed = True

            if not progressed:
                raise ValueError("Dependencies tidak valid: ada siklus pada konfigurasi tahapan.")

        for stage_name in stage_order:
            deps = self.stages[stage_name].dependencies

            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)

            end_times[stage_name] = start_times[stage_name] + results[stage_name]

        results["Total_Duration"] = end_times.max(axis=1)

        for stage_name in self.stages.keys():
            results[f"{stage_name}_Finish"] = end_times[stage_name]
            results[f"{stage_name}_Start"] = start_times[stage_name]

        self.simulation_results = results
        return results

    def calculate_critical_path_probability(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")

        critical_path_probs = {}
        total_duration = self.simulation_results["Total_Duration"]

        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f"{stage_name}_Finish"]
            stage_duration = self.simulation_results[stage_name]

            slack = total_duration - stage_finish
            is_critical = slack < 0.5
            prob_critical = np.mean(is_critical)
            correlation = stage_duration.corr(total_duration)

            critical_path_probs[stage_name] = {
                "probability": prob_critical,
                "correlation": correlation,
                "avg_duration": stage_duration.mean(),
                "avg_slack": slack.mean(),
            }

        return pd.DataFrame(critical_path_probs).T

    def analyze_risk_contribution(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")

        total_var = self.simulation_results["Total_Duration"].var()
        contributions = {}

        for stage_name in self.stages.keys():
            stage_var = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results["Total_Duration"]
            )
            contribution = (stage_covar / total_var) * 100 if total_var > 0 else 0

            contributions[stage_name] = {
                "variance": stage_var,
                "contribution_percent": contribution,
                "std_dev": np.sqrt(stage_var),
            }

        return pd.DataFrame(contributions).T


# ============================================================================
# 3. FUNGSI VISUALISASI PLOTLY
# ============================================================================
def create_distribution_plot(results):
    total_duration = results["Total_Duration"]
    mean_duration = total_duration.mean()
    median_duration = np.median(total_duration)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=total_duration,
            nbinsx=60,
            name="Distribusi Durasi",
            marker_color="skyblue",
            opacity=0.75,
            histnorm="probability density",
        )
    )

    fig.add_vline(
        x=mean_duration,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_duration:.1f} bulan",
    )
    fig.add_vline(
        x=median_duration,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_duration:.1f} bulan",
    )

    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])

    fig.add_vrect(
        x0=ci_80[0],
        x1=ci_80[1],
        fillcolor="yellow",
        opacity=0.2,
        annotation_text="80% CI",
        line_width=0,
    )
    fig.add_vrect(
        x0=ci_95[0],
        x1=ci_95[1],
        fillcolor="orange",
        opacity=0.1,
        annotation_text="95% CI",
        line_width=0,
    )

    fig.update_layout(
        title="Distribusi Durasi Total Proyek",
        xaxis_title="Durasi Total Proyek (Bulan)",
        yaxis_title="Densitas Probabilitas",
        showlegend=True,
        height=500,
    )

    return fig, {
        "mean": mean_duration,
        "median": median_duration,
        "std": total_duration.std(),
        "min": total_duration.min(),
        "max": total_duration.max(),
        "ci_80": ci_80,
        "ci_95": ci_95,
    }


def create_completion_probability_plot(results):
    deadlines = np.arange(14, 26.1, 0.1)
    completion_probs = []

    for deadline in deadlines:
        prob = np.mean(results["Total_Duration"] <= deadline)
        completion_probs.append(prob)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=deadlines,
            y=completion_probs,
            mode="lines",
            name="Probabilitas Selesai",
            line=dict(color="darkblue", width=3),
            fill="tozeroy",
            fillcolor="rgba(173, 216, 230, 0.3)",
        )
    )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="50%",
        annotation_position="right",
    )
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="green",
        annotation_text="70%",
        annotation_position="right",
    )
    fig.add_hline(
        y=0.9,
        line_dash="dash",
        line_color="blue",
        annotation_text="90%",
        annotation_position="right",
    )

    fig.add_vrect(
        x0=18,
        x1=22,
        fillcolor="orange",
        opacity=0.1,
        annotation_text="Deadline Realistis",
        line_width=0,
    )

    key_deadlines = [16, 18, 20, 22, 24]
    for dl in key_deadlines:
        idx = np.argmin(np.abs(deadlines - dl))
        prob = completion_probs[idx]
        fig.add_trace(
            go.Scatter(
                x=[dl],
                y=[prob],
                mode="markers+text",
                marker=dict(size=10, color="red", line=dict(color="black", width=1)),
                text=[f"{dl} bln\n{prob:.1%}"],
                textposition="top center",
                showlegend=False,
            )
        )

    # Titik kapan probabilitas 70%-90% tercapai (inverse dari kurva)
    target_probs = [0.7, 0.8, 0.9]
    target_colors = ["green", "orange", "blue"]
    probs_arr = np.array(completion_probs)
    for p_target, color in zip(target_probs, target_colors):
        month_target = np.interp(p_target, probs_arr, deadlines)
        actual_prob = np.mean(results["Total_Duration"] <= month_target)
        fig.add_trace(
            go.Scatter(
                x=[month_target],
                y=[actual_prob],
                mode="markers+text",
                marker=dict(size=11, color=color, line=dict(color="black", width=1)),
                text=[f"{int(p_target*100)}% ≈ {month_target:.2f} bln"],
                textposition="bottom right",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Kurva Probabilitas Penyelesaian Proyek",
        xaxis_title="Deadline (Bulan)",
        yaxis_title="Probabilitas Selesai Tepat Waktu",
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[14, 26],
        height=500,
    )

    return fig


def create_critical_path_plot(critical_analysis):
    critical_analysis = critical_analysis.sort_values("probability", ascending=True)

    colors = ["red" if prob > 0.7 else "lightcoral" for prob in critical_analysis["probability"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[stage.replace("_", " ") for stage in critical_analysis.index],
            x=critical_analysis["probability"],
            orientation="h",
            marker_color=colors,
            text=[f"{prob:.1%}" for prob in critical_analysis["probability"]],
            textposition="auto",
        )
    )

    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange")

    fig.update_layout(
        title="Analisis Critical Path per Tahapan",
        xaxis_title="Probabilitas Menjadi Critical Path",
        xaxis_range=[0, 1.0],
        height=500,
    )

    return fig


def create_stage_boxplot(results, stages):
    stage_names = list(stages.keys())

    fig = go.Figure()

    for i, stage in enumerate(stage_names):
        data = results[stage]
        fig.add_trace(
            go.Box(
                y=data,
                name=stage.split("_", 1)[1] if "_" in stage else stage,
                boxmean="sd",
                marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                boxpoints="outliers",
                jitter=0.25,
                pointpos=-1.7,
            )
        )

    fig.update_layout(
        title="Distribusi Durasi per Tahapan",
        yaxis_title="Durasi (Bulan)",
        height=500,
        showlegend=False,
    )

    return fig


def create_risk_contribution_plot(risk_contrib):
    risk_contrib = risk_contrib.sort_values("contribution_percent", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[name.split("_", 1)[1] if "_" in name else name for name in risk_contrib.index],
            y=risk_contrib["contribution_percent"],
            marker_color=px.colors.qualitative.Set3,
            text=[f"{contrib:.1f}%" for contrib in risk_contrib["contribution_percent"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Kontribusi Risiko per Tahapan",
        yaxis_title="Kontribusi terhadap Variabilitas (%)",
        height=400,
    )

    return fig


def create_correlation_heatmap(results, stages):
    correlation_matrix = results[list(stages.keys())].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=[name.split("_", 1)[1] if "_" in name else name for name in correlation_matrix.columns],
            y=[name.split("_", 1)[1] if "_" in name else name for name in correlation_matrix.index],
            colorscale="RdBu",
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Matriks Korelasi Antar Tahapan",
        height=500,
    )

    return fig


# ============================================================================
# 4. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    st.markdown(
        '<h1 class="main-header">🏗️ Simulasi Monte Carlo - Proyek Pembangunan Gedung FITE 5 Lantai</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    Aplikasi ini mensimulasikan ketidakpastian durasi proyek pembangunan Gedung FITE menggunakan metode Monte Carlo.
    Hasil utama meliputi distribusi durasi total, probabilitas penyelesaian terhadap deadline,
    analisis tahapan kritis, dan kontribusi risiko setiap tahapan.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
    <div class="creator-card">
        <div class="creator-title">👨‍💻 MAKER</div>
        <div class="creator-name">Jose Mourinho Napitupulu</div>
        <div class="creator-role">Project Simulation Developer</div>
        <div class="creator-link"><a href="https://github.com/JoseNapitupulu" target="_blank">🔗 Kunjungi GitHub</a></div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")

    num_simulations = st.sidebar.slider(
        "🔁 Jumlah Iterasi Simulasi",
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
    )

    # Konfigurasi default tahapan proyek gedung FITE (satuan: bulan)
    default_config = {
        "1_Persiapan_Lahan_dan_Fondasi": {
            "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
            "risk_factors": {
                "cuaca_buruk": {"type": "discrete", "probability": 0.35, "impact": 0.4},
                "kondisi_tanah_sulit": {"type": "discrete", "probability": 0.2, "impact": 0.5},
                "produktivitas_kerja": {"type": "continuous", "mean": 1.0, "std": 0.15},
            },
        },
        "2_Struktur_Beton_5_Lantai": {
            "base_params": {"optimistic": 4.0, "most_likely": 6.0, "pessimistic": 9.0},
            "risk_factors": {
                "keterlambatan_material_beton": {"type": "discrete", "probability": 0.3, "impact": 0.35},
                "cuaca_ekstrem": {"type": "discrete", "probability": 0.25, "impact": 0.3},
                "kesalahan_struktur_minor": {"type": "discrete", "probability": 0.15, "impact": 0.2},
                "efisiensi_kerja_tim": {"type": "continuous", "mean": 1.0, "std": 0.2},
            },
            "dependencies": ["1_Persiapan_Lahan_dan_Fondasi"],
        },
        "3_Selubung_Bangunan_dan_Atap": {
            "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
            "risk_factors": {
                "keterlambatan_material_atap": {"type": "discrete", "probability": 0.25, "impact": 0.3},
                "cuaca_tidak_kondusif": {"type": "discrete", "probability": 0.3, "impact": 0.25},
            },
            "dependencies": ["2_Struktur_Beton_5_Lantai"],
        },
        "4_Sistem_MEP_Basic": {
            "base_params": {"optimistic": 2.0, "most_likely": 3.5, "pessimistic": 5.5},
            "risk_factors": {
                "keterlambatan_peralatan_teknis": {"type": "discrete", "probability": 0.4, "impact": 0.4},
                "ketersediaan_teknisi_ahli": {"type": "continuous", "mean": 1.0, "std": 0.25},
            },
            "dependencies": ["3_Selubung_Bangunan_dan_Atap"],
        },
        "5_Interior_dan_Finishing": {
            "base_params": {"optimistic": 2.0, "most_likely": 3.0, "pessimistic": 5.0},
            "risk_factors": {
                "perubahan_desain": {"type": "discrete", "probability": 0.35, "impact": 0.4},
                "keterlambatan_material_finishing": {"type": "discrete", "probability": 0.3, "impact": 0.25},
            },
            "dependencies": ["4_Sistem_MEP_Basic"],
        },
        "6_Instalasi_Laboratorium_dan_Furnitur": {
            "base_params": {"optimistic": 2.5, "most_likely": 4.0, "pessimistic": 6.5},
            "risk_factors": {
                "keterlambatan_lab_equipment": {"type": "discrete", "probability": 0.45, "impact": 0.5},
                "kompleksitas_instalasi_lab": {"type": "continuous", "mean": 1.0, "std": 0.3},
                "perubahan_spesifikasi_lab": {"type": "discrete", "probability": 0.25, "impact": 0.3},
            },
            "dependencies": ["5_Interior_dan_Finishing"],
        },
        "7_Testing_dan_Commissioning": {
            "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 3.0},
            "risk_factors": {
                "defect_ditemukan": {"type": "discrete", "probability": 0.4, "impact": 0.5},
                "lab_equipment_malfunction": {"type": "discrete", "probability": 0.3, "impact": 0.6},
            },
            "dependencies": ["6_Instalasi_Laboratorium_dan_Furnitur"],
        },
        "8_Final_Inspection_dan_Handover": {
            "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 2.0},
            "risk_factors": {
                "perbaikan_catatan_defect": {"type": "discrete", "probability": 0.3, "impact": 0.4}
            },
            "dependencies": ["7_Testing_dan_Commissioning"],
        },
    }

    st.sidebar.markdown("### 📋 Konfigurasi Tahapan (Bulan)")

    for stage_name, config in default_config.items():
        stage_label = stage_name.split("_", 1)[1] if "_" in stage_name else stage_name
        with st.sidebar.expander(stage_label, expanded=False):
            optimistic = st.number_input(
                "Optimistic",
                min_value=0.1,
                max_value=24.0,
                value=float(config["base_params"]["optimistic"]),
                step=0.1,
                key=f"opt_{stage_name}",
            )
            most_likely = st.number_input(
                "Most Likely",
                min_value=0.1,
                max_value=24.0,
                value=float(config["base_params"]["most_likely"]),
                step=0.1,
                key=f"ml_{stage_name}",
            )
            pessimistic = st.number_input(
                "Pessimistic",
                min_value=0.1,
                max_value=24.0,
                value=float(config["base_params"]["pessimistic"]),
                step=0.1,
                key=f"pes_{stage_name}",
            )

            default_config[stage_name]["base_params"] = {
                "optimistic": optimistic,
                "most_likely": most_likely,
                "pessimistic": pessimistic,
            }

    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div class="muted-text" style="font-size: 0.8rem;">
    <b>ℹ️ Keterangan:</b><br>
    • Optimistic: Estimasi terbaik<br>
    • Most Likely: Estimasi realistis<br>
    • Pessimistic: Estimasi terburuk<br>
    • CI: Confidence Interval<br>
    • Satuan waktu: Bulan
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "simulator" not in st.session_state:
        st.session_state.simulator = None

    if run_simulation:
        np.random.seed(42)
        with st.spinner("⏳ Menjalankan simulasi Monte Carlo..."):
            simulator = MonteCarloProjectSimulation(
                stages_config=default_config,
                num_simulations=num_simulations,
            )
            results = simulator.run_simulation()
            st.session_state.simulation_results = results
            st.session_state.simulator = simulator
            st.success(f"✅ Simulasi selesai: {num_simulations:,} iterasi.")

    if st.session_state.simulation_results is None:
        st.markdown(
            """
        <div class="empty-state-card">
            <h3>🚀 Siap memulai simulasi?</h3>
            <p>Atur parameter di sidebar lalu klik <b>Run Simulation</b> untuk memulai analisis.</p>
            <p>📊 Hasil simulasi akan ditampilkan di sini setelah proses selesai.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<h2 class="sub-header">📋 Preview Konfigurasi Tahapan</h2>', unsafe_allow_html=True)
        for stage_name, config in default_config.items():
            base = config["base_params"]
            stage_label = stage_name.split("_", 1)[1] if "_" in stage_name else stage_name
            st.markdown(
                f"""
            <div class="stage-card">
            <b>{stage_label}</b> |
            Optimistic: {base['optimistic']:.1f} bulan |
            Most Likely: {base['most_likely']:.1f} bulan |
            Pessimistic: {base['pessimistic']:.1f} bulan
            </div>
            """,
                unsafe_allow_html=True,
            )
        return

    results = st.session_state.simulation_results
    simulator = st.session_state.simulator

    # ====================================================================
    # BAGIAN 1: STATISTIK UTAMA
    # ====================================================================
    st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)

    total_duration = results["Total_Duration"]
    mean_duration = total_duration.mean()
    median_duration = np.median(total_duration)
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>{mean_duration:.2f}</h3>
            <p>Rata-rata Durasi (Bulan)</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>{median_duration:.2f}</h3>
            <p>Median Durasi (Bulan)</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>{ci_80[0]:.2f} - {ci_80[1]:.2f}</h3>
            <p>80% Confidence Interval</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>{ci_95[0]:.2f} - {ci_95[1]:.2f}</h3>
            <p>95% Confidence Interval</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ====================================================================
    # BAGIAN 2: VISUALISASI UTAMA
    # ====================================================================
    st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Distribusi Durasi",
            "🎯 Probabilitas Penyelesaian",
            "🔍 Analisis Tahapan",
            "⚠️ Analisis Risiko",
        ]
    )

    with tab1:
        fig_dist, stats = create_distribution_plot(results)
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("📋 Detail Statistik Distribusi"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Statistik Deskriptif:**")
                st.write(f"- Mean: {stats['mean']:.2f} bulan")
                st.write(f"- Median: {stats['median']:.2f} bulan")
                st.write(f"- Std Dev: {stats['std']:.2f} bulan")
                st.write(f"- Min: {stats['min']:.2f} bulan")
                st.write(f"- Max: {stats['max']:.2f} bulan")
            with c2:
                st.write("**Confidence Intervals:**")
                st.write(f"- 80% CI: [{stats['ci_80'][0]:.2f}, {stats['ci_80'][1]:.2f}] bulan")
                st.write(f"- 95% CI: [{stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f}] bulan")

    with tab2:
        fig_prob = create_completion_probability_plot(results)
        st.plotly_chart(fig_prob, use_container_width=True)

        with st.expander("📅 Analisis Probabilitas Deadline"):
            deadlines = [16, 18, 20, 22, 24]
            cols = st.columns(len(deadlines))
            for i, deadline in enumerate(deadlines):
                prob_on_time = np.mean(total_duration <= deadline)
                prob_late = 1 - prob_on_time
                with cols[i]:
                    st.metric(
                        label=f"Deadline {deadline} bulan",
                        value=f"{prob_on_time:.1%}",
                        delta=f"{prob_late:.1%} terlambat" if prob_late > 0 else "Tepat waktu",
                        delta_color="inverse",
                    )

            st.markdown("---")
            st.markdown("**📌 Estimasi Bulan Selesai pada Confidence 70%-90%**")
            deadlines_dense = np.arange(14, 26.1, 0.1)
            completion_probs_dense = [np.mean(total_duration <= d) for d in deadlines_dense]
            probs_arr = np.array(completion_probs_dense)

            p70 = np.interp(0.7, probs_arr, deadlines_dense)
            p80 = np.interp(0.8, probs_arr, deadlines_dense)
            p90 = np.interp(0.9, probs_arr, deadlines_dense)

            p70_actual = np.mean(total_duration <= p70)
            p80_actual = np.mean(total_duration <= p80)
            p90_actual = np.mean(total_duration <= p90)

            conf_cols = st.columns(3)
            with conf_cols[0]:
                st.metric(label="Selesai 70%", value=f"{p70:.2f} bulan", delta=f"Prob aktual: {p70_actual:.1%}")
            with conf_cols[1]:
                st.metric(label="Selesai 80%", value=f"{p80:.2f} bulan", delta=f"Prob aktual: {p80_actual:.1%}")
            with conf_cols[2]:
                st.metric(label="Selesai 90%", value=f"{p90:.2f} bulan", delta=f"Prob aktual: {p90_actual:.1%}")

    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            critical_analysis = simulator.calculate_critical_path_probability()
            fig_critical = create_critical_path_plot(critical_analysis)
            st.plotly_chart(fig_critical, use_container_width=True)

        with col_b:
            fig_boxplot = create_stage_boxplot(results, simulator.stages)
            st.plotly_chart(fig_boxplot, use_container_width=True)

        with st.expander("🧭 Detail Analisis Critical Path"):
            critical_df = critical_analysis.sort_values("probability", ascending=False)
            st.dataframe(critical_df, use_container_width=True)

    with tab4:
        col_a, col_b = st.columns(2)

        with col_a:
            risk_contrib = simulator.analyze_risk_contribution()
            fig_risk = create_risk_contribution_plot(risk_contrib)
            st.plotly_chart(fig_risk, use_container_width=True)

        with col_b:
            fig_corr = create_correlation_heatmap(results, simulator.stages)
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("🧪 Detail Analisis Kontribusi Risiko"):
            st.dataframe(risk_contrib, use_container_width=True)

    # ====================================================================
    # BAGIAN 3: ANALISIS STATISTIK LENGKAP
    # ====================================================================
    st.markdown('<h2 class="sub-header">📋 Analisis Statistik Lengkap</h2>', unsafe_allow_html=True)

    with st.expander("📊 Tabel Data Simulasi", expanded=False):
        st.caption(f"Menampilkan seluruh data simulasi: {len(results):,} baris")
        st.dataframe(results, use_container_width=True, height=420)
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Data Simulasi (CSV)",
            data=csv_data,
            file_name="hasil_simulasi_proyek_gedung_fite.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("**📐 Statistik Durasi per Tahapan (Bulan):**")
    stage_stats = pd.DataFrame()
    for stage_name in simulator.stages.keys():
        stage_data = results[stage_name]
        stage_stats[stage_name] = [
            stage_data.mean(),
            stage_data.std(),
            np.percentile(stage_data, 25),
            np.percentile(stage_data, 50),
            np.percentile(stage_data, 75),
        ]

    stage_stats.index = ["Mean", "Std Dev", "Q1", "Median", "Q3"]
    stage_stats_display = stage_stats.T.copy()
    stage_stats_display.index = [
        idx.split("_", 1)[1] if "_" in idx else idx for idx in stage_stats_display.index
    ]
    st.dataframe(stage_stats_display, use_container_width=True)

    # ====================================================================
    # BAGIAN 4: ANALISIS DEADLINE DAN REKOMENDASI
    # ====================================================================
    st.markdown('<h2 class="sub-header">🎯 Analisis Deadline dan Rekomendasi</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        target_deadline = st.number_input(
            "📌 Masukkan deadline target (bulan):",
            min_value=12.0,
            max_value=30.0,
            value=20.0,
            step=0.5,
        )

        prob_target = np.mean(total_duration <= target_deadline)
        delay_risk = max(0.0, np.percentile(total_duration, 95) - target_deadline)

        st.metric(
            label=f"Probabilitas selesai dalam {target_deadline:.1f} bulan",
            value=f"{prob_target:.1%}",
            delta=f"Potensi keterlambatan: {delay_risk:.2f} bulan" if delay_risk > 0 else "Tepat waktu",
            delta_color="inverse",
        )

    with col2:
        safety_buffer_80 = np.percentile(total_duration, 80) - mean_duration
        contingency_95 = np.percentile(total_duration, 95) - mean_duration

        st.markdown(
            f"""
        <div class="info-box">
            <h4>Rekomendasi Manajemen Risiko</h4>
            • Safety Buffer (80%): <b>{safety_buffer_80:.2f} bulan</b><br>
            • Contingency Reserve (95%): <b>{contingency_95:.2f} bulan</b><br><br>
            • Estimasi jadwal rekomendasi:<br>
            <b>{mean_duration:.2f} + {safety_buffer_80:.2f} = {mean_duration + safety_buffer_80:.2f} bulan</b>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with st.expander("🗂️ Data Simulasi (Seluruh Baris)", expanded=False):
        st.caption(f"Total data simulasi: {len(results):,} baris")
        st.dataframe(results, use_container_width=True, height=450)

    with st.expander("ℹ️ Informasi Teknis Simulasi", expanded=False):
        st.write("**Parameter Simulasi:**")
        st.write(f"- Jumlah iterasi: {num_simulations:,}")
        st.write(f"- Jumlah tahapan: {len(simulator.stages)}")
        st.write("- Seed acak: 42 (untuk hasil yang dapat direproduksi)")

        st.write("\n**Ringkasan Statistik Durasi Total:**")
        st.write(f"- Mean: {mean_duration:.2f} bulan")
        st.write(f"- Median: {median_duration:.2f} bulan")
        st.write(f"- Std Dev: {total_duration.std():.2f} bulan")
        st.write(f"- Min - Max: {total_duration.min():.2f} - {total_duration.max():.2f} bulan")
        st.write(
            f"- 80% CI: [{np.percentile(total_duration, 10):.2f}, {np.percentile(total_duration, 90):.2f}] bulan"
        )
        st.write(
            f"- 95% CI: [{np.percentile(total_duration, 2.5):.2f}, {np.percentile(total_duration, 97.5):.2f}] bulan"
        )

        st.write("\n**Estimasi Bulan Penyelesaian Berdasarkan Confidence:**")
        st.write(f"- 50% selesai: {np.percentile(total_duration, 50):.2f} bulan")
        st.write(f"- 70% selesai: {np.percentile(total_duration, 70):.2f} bulan")
        st.write(f"- 80% selesai: {np.percentile(total_duration, 80):.2f} bulan")
        st.write(f"- 90% selesai: {np.percentile(total_duration, 90):.2f} bulan")
        st.write(f"- 95% selesai: {np.percentile(total_duration, 95):.2f} bulan")

        st.write("\n**Top 3 Tahapan Paling Kritis:**")
        critical_info = simulator.calculate_critical_path_probability().sort_values(
            "probability", ascending=False
        )
        for i, (stage_name, row) in enumerate(critical_info.head(3).iterrows(), 1):
            stage_label = stage_name.split("_", 1)[1] if "_" in stage_name else stage_name
            st.write(
                f"{i}. {stage_label} | Critical: {row['probability']:.1%} | Avg durasi: {row['avg_duration']:.2f} bulan"
            )

        st.write("\n**Top 3 Kontributor Risiko:**")
        risk_info = simulator.analyze_risk_contribution().sort_values(
            "contribution_percent", ascending=False
        )
        for i, (stage_name, row) in enumerate(risk_info.head(3).iterrows(), 1):
            stage_label = stage_name.split("_", 1)[1] if "_" in stage_name else stage_name
            st.write(
                f"{i}. {stage_label} | Kontribusi: {row['contribution_percent']:.2f}% | Std dev: {row['std_dev']:.3f} bulan"
            )

        st.write("\n**Konfigurasi Tahapan:**")
        for stage_name, config in default_config.items():
            base = config["base_params"]
            stage_label = stage_name.split("_", 1)[1] if "_" in stage_name else stage_name
            st.markdown(
                f"""
            <div class="stage-card">
            <b>{stage_label}</b><br>
            • Optimistic: {base['optimistic']:.1f} bulan<br>
            • Most Likely: {base['most_likely']:.1f} bulan<br>
            • Pessimistic: {base['pessimistic']:.1f} bulan
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; color:#666; font-size:0.9rem;">
    <p><b>🏗️ Simulasi Monte Carlo - Proyek Gedung FITE</b></p>
    <p>⚠️ Hasil simulasi adalah estimasi probabilistik, bukan prediksi pasti.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 5. JALANKAN APLIKASI
# ============================================================================
if __name__ == "__main__":
    main()
