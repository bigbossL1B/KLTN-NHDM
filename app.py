from datetime import date, timedelta
from pathlib import Path
import base64
import mimetypes

import pandas as pd
import streamlit as st
import plotly.express as px

from data.industry_tickers import INDUSTRY_TICKERS
from utils.data_loader import load_price_data, make_price_pivot, calculate_returns
from utils.portfolio import (
    select_top_by_sharpe,
    build_allocation_equal,
    build_allocation_80_20,
    port_char,
    port_char_from_series,
    sharpe_port,
    sharpe_from_series,
)
from utils.modeling import prepare_train_test, train_multi_seed


# ================= CONFIG =================
RF_ANNUAL = 0.045
TRADING_DAYS = 252
TOP_N = 10
WINDOW_SIZE = 30
EPOCHS = 100
BATCH_SIZE = 32
SEED_LIST = [7, 21, 42, 99, 123]

st.set_page_config(
    page_title="Portfolio Optimization App",
    page_icon="📈",
    layout="wide",
)


# ================= IMAGE HELPER =================
def image_to_base64(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{data}"


logo_src = image_to_base64("assets/hub_logo.png")


# ================= CSS =================
st.markdown(
    """
<style>
/* ================= GLOBAL ================= */
.stApp {
    background: linear-gradient(rgba(255,255,255,.90), rgba(255,255,255,.94)),
                url('https://images.unsplash.com/photo-1563986768494-4dee2763ff3f?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-attachment: fixed;
}

.block-container {
    max-width: 1240px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* Sidebar nhìn sạch hơn */
section[data-testid="stSidebar"] {
    background: #f1f4f8;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #283142;
}

/* ================= HERO ================= */
.hero {
    width: 100%;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    justify-content: center;

    padding: 48px 70px;
    border-radius: 28px;

    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
    box-shadow: 0 18px 48px rgba(31, 45, 61, 0.13);
    margin-bottom: 28px;
}

/* Logo + tên trường nằm ngang */
.brand-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 26px;
    margin-bottom: 22px;
}

.logo {
    width: 150px;
    height: auto;
    display: block;
}

.school {
    color: #155b93;
    font-weight: 800;
    font-size: 25px;
    line-height: 1.35;
    letter-spacing: .2px;
    text-transform: uppercase;
}

/* đường kẻ trang trí */
.hero-divider {
    width: 520px;
    max-width: 80%;
    margin: 4px auto 22px auto;
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-divider::before,
.hero-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #aab7c8, transparent);
}

.hero-dot {
    width: 9px;
    height: 9px;
    background: #0d78c8;
    transform: rotate(45deg);
    border-radius: 2px;
}

/* 2 phần bạn khoanh đỏ căn giữa */
.app-title {
    max-width: 860px;
    margin: 0 auto;
    text-align: center;

    font-size: 39px;
    font-weight: 850;
    color: #2d3342;
    line-height: 1.28;
    letter-spacing: -0.8px;
}

.guide-wrap {
    text-align: center;
    margin-top: 20px;
}

.guide {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;

    color: #2f8a3a;
    font-weight: 800;
    font-size: 24px;
    margin-bottom: 14px;
}

.guide-text {
    max-width: 760px;
    margin: 0 auto;
    text-align: center;

    color: #343b48;
    font-size: 16px;
    line-height: 1.85;
}

.guide-text .steps {
    margin-top: 14px;
    display: inline-block;
    text-align: left;
}

.guide-text b {
    color: #1f2937;
}

code {
    background: rgba(46,125,50,.11);
    color: #1b7f3a;
    padding: 3px 8px;
    border-radius: 8px;
    font-weight: 700;
}

/* Info box */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Metric card */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.82);
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,.06);
}

/* Responsive */
@media (max-width: 900px) {
    .hero {
        padding: 34px 28px;
    }

    .brand-row {
        flex-direction: column;
        gap: 14px;
    }

    .logo {
        width: 125px;
    }

    .school {
        text-align: center;
        font-size: 20px;
    }

    .app-title {
        font-size: 30px;
    }

    .guide {
        font-size: 21px;
    }

    .guide-text {
        font-size: 15px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Thiết lập")

    api_key = st.text_input(
        "VNStock API key",
        type="password",
        help="Dán API key từ vnstocks.com để tăng giới hạn request. Bỏ trống thì app vẫn chạy theo giới hạn Guest.",
    )

    industry = st.selectbox(
        "Chọn ngành",
        list(INDUSTRY_TICKERS.keys()),
        index=list(INDUSTRY_TICKERS.keys()).index("Thép")
        if "Thép" in INDUSTRY_TICKERS
        else 0,
    )

    st.markdown("### 📅 Khoảng thời gian phân tích")

    start_date = st.date_input("Ngày bắt đầu", value=date(2015, 1, 1))
    end_date = st.date_input("Ngày kết thúc", value=date(2025, 12, 31))

    test_start = date(end_date.year, 1, 1)
    test_end = end_date
    train_start = start_date
    train_end = test_start - timedelta(days=1)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    run_btn = st.button("🚀 Bắt đầu", use_container_width=True)


# ================= HEADER =================
logo_html = (
    f'<img src="{logo_src}" class="logo" alt="HUB Logo">'
    if logo_src
    else '<div style="font-size:72px;">🏦</div>'
)

st.markdown(
    f"""
<div class="hero">
<div class="brand-row">
{logo_html}
<div class="school">
TRƯỜNG ĐẠI HỌC NGÂN HÀNG<br>
THÀNH PHỐ HỒ CHÍ MINH
</div>
</div>

<div class="hero-divider">
<div class="hero-dot"></div>
</div>

<div class="app-title">
Xây dựng danh mục đầu tư tối ưu<br>
bằng mô hình LSTM - GRU
</div>

<div class="hero-divider" style="margin-top:24px;">
<div class="hero-dot"></div>
</div>

<div class="guide-wrap">
<div class="guide">📘 Hướng dẫn sử dụng</div>
<div class="guide-text">
Ứng dụng có 2 cách nhập dữ liệu:
<br>
<div class="steps">
<b>1.</b> Chọn ngành + thời gian → hệ thống tự động lấy dữ liệu, train model và đánh giá.<br>
<b>2.</b> Upload CSV với các cột:
<code>time</code>, <code>ticker</code>, <code>close</code>
</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# ================= VALIDATE =================
if not run_btn:
    st.info("Bấm **Bắt đầu** để tải dữ liệu, xây dựng mô hình và xem kết quả.")
    st.stop()

if start_date >= end_date:
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()

if start_date >= test_start:
    st.error("Khoảng thời gian chưa đủ để xây dựng mô hình. Hãy chọn ngày bắt đầu trước năm kết thúc ít nhất 1 năm.")
    st.stop()


st.info(
    f"📌 Hệ thống sẽ dùng dữ liệu từ **{train_start:%Y/%m/%d} → {train_end:%Y/%m/%d}** "
    f"để xây dựng mô hình và đánh giá hiệu quả trong giai đoạn "
    f"**{test_start:%Y/%m/%d} → {test_end:%Y/%m/%d}**."
)


# ================= PROGRESS =================
status_box = st.empty()
progress_bar = st.progress(0)
log_box = st.empty()
logs = []


def ui_progress(idx, total, ticker, status, detail):
    pct = min(idx / max(total, 1), 1.0)
    progress_bar.progress(pct)

    icon = {
        "loading": "🔄",
        "ok": "✅",
        "failed": "❌",
        "retry": "⚠️",
        "rate_limit": "⏳",
        "cooldown": "😴",
        "cache": "💾",
    }.get(status, "•")

    line = f"{icon} [{idx}/{total}] {ticker}: {detail or status}"
    status_box.info(line)
    logs.append(line)
    log_box.code("\n".join(logs[-12:]), language="text")


# ================= LOAD DATA =================
with st.spinner("Đang tải dữ liệu và xử lý..."):
    if uploaded is not None:
        raw_data = pd.read_csv(uploaded)

        required = {"time", "ticker", "close"}
        if not required.issubset(raw_data.columns):
            st.error("CSV phải có đủ cột: time, ticker, close.")
            st.stop()

        failed = []
    else:
        tickers = sorted(list(set(INDUSTRY_TICKERS[industry])))

        raw_data, failed = load_price_data(
            tickers,
            train_start,
            test_end,
            api_key=api_key,
            progress_callback=ui_progress,
        )

    if raw_data.empty:
        st.error("Không tải được dữ liệu.")
        st.stop()

    price_pivot = make_price_pivot(raw_data)
    returns_df = calculate_returns(price_pivot)


progress_bar.progress(1.0)
status_box.success("Đã tải/xử lý dữ liệu xong.")

st.caption(
    f"Khoảng dữ liệu tải được: {price_pivot.index.min().date()} → {price_pivot.index.max().date()} | "
    f"Returns: {returns_df.index.min().date()} → {returns_df.index.max().date()} | "
    f"Số dòng returns: {len(returns_df)}"
)

if failed:
    with st.expander("Một số mã không tải được"):
        st.dataframe(pd.DataFrame(failed), use_container_width=True)


# ================= SHARPE =================
if returns_df.shape[1] < TOP_N:
    st.error(f"Không đủ mã sau xử lý. Cần ít nhất {TOP_N}, hiện có {returns_df.shape[1]}.")
    st.stop()

selected_symbols, sharpe_table = select_top_by_sharpe(returns_df, top_n=TOP_N)

st.subheader("📌 Top 10 mã cổ phiếu có Sharpe Ratio cao nhất")
st.write(", ".join(selected_symbols))

m1, m2, m3 = st.columns(3)
m1.metric("Số mã ban đầu", price_pivot.shape[1])
m2.metric("Số mã sau dropna", returns_df.shape[1])
m3.metric("Số mã train", len(selected_symbols))

with st.expander("Xem Sharpe Ratio từng mã"):
    st.dataframe(
        sharpe_table.reset_index().rename(columns={"index": "ticker", 0: "Sharpe"}),
        use_container_width=True,
    )


# ================= PREPARE TRAIN/TEST =================
try:
    (
        X_train,
        y_train,
        train_dates,
        X_test,
        y_test,
        test_dates,
        train_returns,
        test_returns,
        scaler,
        debug_shapes,
    ) = prepare_train_test(
        price_pivot,
        selected_symbols,
        train_start,
        train_end,
        test_start,
        test_end,
        window_size=WINDOW_SIZE,
    )
except Exception as e:
    st.error(f"Lỗi tạo dữ liệu mô hình: {e}")
    st.warning("Cách xử lý nhanh: chọn khoảng ngày có đủ dữ liệu hơn hoặc kiểm tra các mã bị thiếu dữ liệu.")
    st.stop()

st.info(
    f"Dữ liệu mô hình: giai đoạn học X={X_train.shape}, y={y_train.shape} | "
    f"giai đoạn đánh giá X={X_test.shape}, y={y_test.shape}"
)

with st.expander("Thông tin kỹ thuật về dữ liệu mô hình"):
    st.json({k: str(v) for k, v in debug_shapes.items()})


# ================= TRAIN MODEL =================
try:
    with st.spinner("Đang xây dựng mô hình LSTM-GRU nhiều lần để chọn kết quả tốt nhất, bước này có thể hơi lâu..."):
        best, runs_df = train_multi_seed(
            X_train,
            y_train,
            X_test,
            y_test,
            test_dates,
            selected_symbols,
            seed_list=SEED_LIST,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
except Exception as e:
    st.error(f"Train model bị lỗi: {e}")
    st.stop()

st.success(f"Mô hình tốt nhất: lần chạy #{best['seed']} | Sharpe LSTM-GRU Dynamic: {best['sharpe']:.4f}")

with st.expander("Kết quả từng lần chạy mô hình"):
    st.dataframe(
        runs_df.style.format(
            {
                "Lợi nhuận năm (%)": "{:.2f}",
                "Rủi ro năm (%)": "{:.2f}",
                "Sharpe": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


# ================= RESULT =================
weights_lstm_avg = best["pred_weights"].mean(axis=0)

results_lstm = (
    pd.DataFrame({"Asset": selected_symbols, "Weight": weights_lstm_avg})
    .sort_values("Weight", ascending=False)
    .reset_index(drop=True)
)

allo_1 = build_allocation_equal(selected_symbols)
allo_2 = build_allocation_80_20(train_returns)

Er_lstm, std_lstm = port_char_from_series(best["portfolio_returns"], annualize=True)
Er_1, std_1 = port_char(allo_1, test_returns, annualize=True)
Er_2, std_2 = port_char(allo_2, test_returns, annualize=True)

sharpe_lstm = sharpe_from_series(best["portfolio_returns"])
sharpe_1 = sharpe_port(allo_1, test_returns)
sharpe_2 = sharpe_port(allo_2, test_returns)

comparison = pd.DataFrame(
    {
        "Chiến lược đầu tư": ["LSTM-GRU (Dynamic)", "Phân bổ đều", "Phân bổ 80-20"],
        "Lợi nhuận trung bình (%)": [Er_lstm * 100, Er_1 * 100, Er_2 * 100],
        "Độ lệch chuẩn (%)": [std_lstm * 100, std_1 * 100, std_2 * 100],
        "Hệ số Sharpe": [sharpe_lstm, sharpe_1, sharpe_2],
    }
)

st.subheader("📊 Kết quả so sánh chiến lược")
st.dataframe(
    comparison.style.format(
        {
            "Lợi nhuận trung bình (%)": "{:.2f}",
            "Độ lệch chuẩn (%)": "{:.2f}",
            "Hệ số Sharpe": "{:.4f}",
        }
    ),
    use_container_width=True,
)


st.subheader("🥧 Tỷ trọng trung bình danh mục LSTM-GRU")
results_lstm["Weight (%)"] = results_lstm["Weight"] * 100

cc1, cc2 = st.columns([1, 1])

with cc1:
    st.dataframe(
        results_lstm.style.format(
            {
                "Weight": "{:.4f}",
                "Weight (%)": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

with cc2:
    fig = px.pie(
        results_lstm,
        names="Asset",
        values="Weight",
        title="Tỷ trọng trung bình LSTM-GRU",
    )
    st.plotly_chart(fig, use_container_width=True)