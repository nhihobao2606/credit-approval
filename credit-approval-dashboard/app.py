import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# CẤU HÌNH
# =========================================================
st.set_page_config(
    page_title="Credit Approval Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("credit_approval_best_model.joblib")


def load_data():
    return pd.read_csv("new_data.csv")


best_model = load_model()
new_data = load_data()

# =========================================================
# TÊN CỘT
# =========================================================
column_names = {
    "A1": "Giới tính",
    "A2": "Tuổi",
    "A3": "Khoản nợ",
    "A4": "Tình trạng hôn nhân",
    "A5": "Khách hàng ngân hàng",
    "A6": "Ngành nghề",
    "A7": "Dân tộc",
    "A8": "Thời gian làm việc",
    "A9": "Nợ quá hạn trước đây",
    "A10": "Đang làm việc",
    "A11": "Điểm tín dụng",
    "A12": "Bằng lái",
    "A13": "Quốc tịch",
    "A14": "Mã vùng",
    "A15": "Thu nhập"
}

model_columns = list(column_names.keys())

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

/* ==============================
   BACKGROUND
   ============================== */

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(99,102,241,0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 100% 0%,
            rgba(236,72,153,0.10),
            transparent 25%
        ),
        #f5f7fb;
}

/* ==============================
   HEADER
   ============================== */

.dashboard-header {
    background:
        linear-gradient(
            135deg,
            #4f46e5 0%,
            #7c3aed 45%,
            #db2777 100%
        );

    padding: 30px 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    color: white;

    box-shadow:
        0 12px 30px rgba(79,70,229,0.25);
}

.dashboard-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.dashboard-subtitle {
    font-size: 14px;
    margin-top: 7px;
    opacity: 0.88;
}

/* ==============================
   BUTTON
   ============================== */

div.stButton > button {
    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;

    padding: 13px 25px !important;

    font-size: 14px !important;
    font-weight: 700 !important;

    box-shadow:
        0 7px 18px rgba(79,70,229,0.25) !important;

    transition: 0.2s !important;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 24px rgba(79,70,229,0.35) !important;
}

/* ==============================
   KPI
   ============================== */

.kpi-card {
    background: rgba(255,255,255,0.95);
    border-radius: 18px;
    padding: 20px;
    min-height: 125px;

    border: 1px solid rgba(226,232,240,0.9);

    box-shadow:
        0 6px 20px rgba(15,23,42,0.06);

    position: relative;
    overflow: hidden;
}

.kpi-card::after {
    content: "";
    position: absolute;

    width: 80px;
    height: 80px;

    right: -30px;
    top: -30px;

    border-radius: 50%;

    background: rgba(99,102,241,0.08);
}

.kpi-icon {
    font-size: 25px;
    margin-bottom: 8px;
}

.kpi-label {
    font-size: 11px;
    font-weight: 700;

    color: #64748b;

    letter-spacing: 0.7px;
}

.kpi-number {
    font-size: 30px;
    font-weight: 800;

    color: #111827;

    margin-top: 2px;
}

.kpi-description {
    font-size: 11px;
    color: #94a3b8;
}

/* ==============================
   SECTION
   ============================== */

.section-title {
    font-size: 20px;
    font-weight: 800;

    color: #111827;

    margin-top: 28px;
    margin-bottom: 5px;
}

.section-subtitle {
    font-size: 13px;
    color: #64748b;

    margin-bottom: 15px;
}

/* ==============================
   INSIGHT
   ============================== */

.insight-card {
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #fdf2f8
        );

    border: 1px solid #e0e7ff;
    border-radius: 16px;

    padding: 18px 22px;
    margin-top: 20px;

    box-shadow:
        0 5px 16px rgba(79,70,229,0.06);
}

.insight-title {
    font-size: 14px;
    font-weight: 800;

    color: #4338ca;

    margin-bottom: 6px;
}

.insight-text {
    font-size: 13px;
    color: #475569;
}

/* ==============================
   TABLE
   ============================== */

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ==============================
   PAGE
   ============================== */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# QUAN TRỌNG: KHÔNG THỤT ĐẦU DÒNG HTML
# =========================================================

st.markdown("""
<div class="dashboard-header">
<div class="dashboard-title">💳 Credit Approval Analytics</div>
<div class="dashboard-subtitle">Machine Learning Dashboard &nbsp;•&nbsp; Phân tích và dự đoán hồ sơ tín dụng</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# NÚT DỰ ĐOÁN
# =========================================================

predict_button = st.button(
    "🔮  DỰ ĐOÁN TOÀN BỘ KHÁCH HÀNG",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # DỮ LIỆU ĐẦU VÀO
    # -----------------------------------------------------

    X_new = new_data[model_columns].copy()

    # Giữ cách xử lý dữ liệu giống code hiện tại
    for col in model_columns:
        X_new[col] = X_new[col].apply(
            lambda x:
                None
                if pd.isna(x)
                else (
                    str(int(x))
                    if isinstance(x, float) and x.is_integer()
                    else str(x)
                )
        )

    # -----------------------------------------------------
    # DỰ ĐOÁN
    # -----------------------------------------------------

    predictions = best_model.predict(X_new)

    # -----------------------------------------------------
    # KẾT QUẢ
    # -----------------------------------------------------

    result = X_new.copy()

    result.insert(
        0,
        "STT",
        range(1, len(result) + 1)
    )

    result["Dự đoán"] = predictions

    result["Kết quả"] = result["Dự đoán"].map({
        "+": "Chấp thuận",
        "-": "Từ chối"
    })

    result = result.rename(
        columns=column_names
    )

    ordered_cols = (
        ["STT"]
        + list(column_names.values())
        + ["Dự đoán", "Kết quả"]
    )

    result = result[ordered_cols]

    # =====================================================
    # KPI
    # =====================================================

    total = len(result)

    approved = int(
        (predictions == "+").sum()
    )

    rejected = int(
        (predictions == "-").sum()
    )

    approved_rate = (
        approved / total * 100
        if total else 0
    )

    rejected_rate = (
        rejected / total * 100
        if total else 0
    )

    # =====================================================
    # TIÊU ĐỀ
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Tổng quan hồ sơ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Tổng hợp kết quả dự đoán từ mô hình học máy'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-icon">👥</div>
<div class="kpi-label">TỔNG HỒ SƠ</div>
<div class="kpi-number">{total}</div>
<div class="kpi-description">Hồ sơ được phân tích</div>
</div>
""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-icon">✅</div>
<div class="kpi-label">CHẤP THUẬN</div>
<div class="kpi-number">{approved}</div>
<div class="kpi-description">{approved_rate:.1f}% tổng hồ sơ</div>
</div>
""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-icon">🚫</div>
<div class="kpi-label">TỪ CHỐI</div>
<div class="kpi-number">{rejected}</div>
<div class="kpi-description">{rejected_rate:.1f}% tổng hồ sơ</div>
</div>
""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-icon">📈</div>
<div class="kpi-label">TỶ LỆ CHẤP THUẬN</div>
<div class="kpi-number">{approved_rate:.1f}%</div>
<div class="kpi-description">Theo kết quả dự đoán</div>
</div>
""", unsafe_allow_html=True)

    # =====================================================
    # BIỂU ĐỒ
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Phân tích kết quả dự đoán</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Trực quan hóa tỷ lệ và số lượng hồ sơ theo kết quả'
        '</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    # =====================================================
    # DONUT
    # =====================================================

    with chart1:

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Chấp thuận",
                        "Từ chối"
                    ],
                    values=[
                        approved,
                        rejected
                    ],
                    hole=0.68,

                    marker=dict(
                        colors=[
                            "#6366F1",
                            "#F472B6"
                        ],
                        line=dict(
                            color="white",
                            width=4
                        )
                    ),

                    textinfo="percent",

                    textfont=dict(
                        size=15,
                        color="white"
                    ),

                    hovertemplate=
                        "<b>%{label}</b><br>"
                        "Số hồ sơ: %{value}<br>"
                        "Tỷ lệ: %{percent}"
                        "<extra></extra>"
                )
            ]
        )

        fig_donut.add_annotation(
            text=f"<b>{total}</b><br><span>Hồ sơ</span>",
            x=0.5,
            y=0.5,
            showarrow=False,

            font=dict(
                size=19,
                color="#1e293b"
            )
        )

        fig_donut.update_layout(
            title=dict(
                text="Tỷ lệ kết quả dự đoán",
                font=dict(
                    size=18,
                    color="#111827"
                ),
                x=0.03
            ),

            height=390,

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),

            legend=dict(
                orientation="h",
                y=-0.02,
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    # =====================================================
    # BAR
    # =====================================================

    with chart2:

        fig_bar = go.Figure()

        fig_bar.add_trace(
            go.Bar(
                x=[
                    "Chấp thuận",
                    "Từ chối"
                ],

                y=[
                    approved,
                    rejected
                ],

                marker=dict(
                    color=[
                        "#6366F1",
                        "#F472B6"
                    ],

                    line=dict(
                        width=0
                    )
                ),

                text=[
                    approved,
                    rejected
                ],

                textposition="outside",

                textfont=dict(
                    size=14,
                    color="#334155"
                ),

                hovertemplate=
                    "<b>%{x}</b><br>"
                    "Số hồ sơ: %{y}"
                    "<extra></extra>"
            )
        )

        fig_bar.update_layout(

            title=dict(
                text="Số lượng hồ sơ theo kết quả",

                font=dict(
                    size=18,
                    color="#111827"
                ),

                x=0.03
            ),

            height=390,

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=55,
                r=25,
                t=60,
                b=50
            ),

            yaxis=dict(
                title="Số lượng hồ sơ",
                gridcolor="#e2e8f0",
                zeroline=False
            ),

            xaxis=dict(
                showgrid=False
            ),

            showlegend=False
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # =====================================================
    # INSIGHT
    # =====================================================

    if approved_rate >= 50:

        insight_text = (
            f"Mô hình dự đoán {approved} hồ sơ được chấp thuận, "
            f"chiếm {approved_rate:.1f}% tổng số hồ sơ. "
            f"Số hồ sơ chấp thuận đang cao hơn số hồ sơ bị từ chối."
        )

    else:

        insight_text = (
            f"Mô hình dự đoán {rejected} hồ sơ bị từ chối, "
            f"chiếm {rejected_rate:.1f}% tổng số hồ sơ. "
            f"Số hồ sơ bị từ chối đang cao hơn số hồ sơ được chấp thuận."
        )

    st.markdown(
        f"""
<div class="insight-card">
<div class="insight-title">💡 INSIGHT</div>
<div class="insight-text">{insight_text}</div>
</div>
""",
        unsafe_allow_html=True
    )

    # =====================================================
    # BẢNG CHI TIẾT
    # =====================================================

    st.markdown(
        '<div class="section-title">👥 Chi tiết hồ sơ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Danh sách hồ sơ và kết quả dự đoán của mô hình'
        '</div>',
        unsafe_allow_html=True
    )

    display_result = result.rename(columns={
        "Tình trạng hôn nhân": "Hôn nhân",
        "Khách hàng ngân hàng": "KH ngân hàng",
        "Thời gian làm việc": "TG làm việc",
        "Nợ quá hạn trước đây": "Nợ quá hạn",
        "Điểm tín dụng": "Điểm tín dụng"
    })

    st.dataframe(
        display_result,
        use_container_width=True,
        hide_index=True,
    )

else:

    # =====================================================
    # TRẠNG THÁI BAN ĐẦU
    # =====================================================

    st.markdown(
        """
<div class="insight-card">
<div class="insight-title">🚀 Sẵn sàng phân tích</div>
<div class="insight-text">
Nhấn nút <b>DỰ ĐOÁN TOÀN BỘ KHÁCH HÀNG</b>
để mô hình phân tích các hồ sơ tín dụng.
</div>
</div>
""",
        unsafe_allow_html=True
    )