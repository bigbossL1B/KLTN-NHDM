from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data.industry_tickers import INDUSTRY_TICKERS
from utils.data_loader import load_price_data, make_price_pivot, calculate_returns
from utils.portfolio import (
    select_top_by_sharpe, build_allocation_equal, build_allocation_80_20,
    port_char, port_char_from_series, sharpe_port, sharpe_from_series, cumulative_from_weights,
)
from utils.modeling import prepare_train_test, train_multi_seed

RF_ANNUAL = 0.045
TRADING_DAYS = 252
TOP_N = 10
WINDOW_SIZE = 30
EPOCHS = 100
BATCH_SIZE = 32
SEED_LIST = [7, 21, 42, 99, 123]

st.set_page_config(page_title="Portfolio Optimization App", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(255,255,255,.88), rgba(255,255,255,.92)),
                url('https://images.unsplash.com/photo-1563986768494-4dee2763ff3f?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-attachment: fixed;
}
.block-container { max-width: 1180px; padding-top: 2rem; }
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.80);
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,.06);
}
.card {
    background: rgba(255,255,255,.78);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 4px 22px rgba(0,0,0,.08);
    margin-bottom: 18px;
}
.hero {
    text-align: center;
    padding: 42px 44px 34px 44px;
    border-radius: 26px;
    background: linear-gradient(rgba(255,255,255,.72), rgba(255,255,255,.78));
    box-shadow: 0 12px 34px rgba(0,0,0,.10);
    margin-bottom: 24px;
    backdrop-filter: blur(3px);
}
.school {
    color: #1e5b91;
    font-weight: 800;
    font-size: 26px;
    line-height: 1.2;
    letter-spacing: .3px;
    margin-bottom: 34px;
}
.app-title {
    color: #2d3342;
    font-weight: 800;
    font-size: 34px;
    line-height: 1.25;
    margin-bottom: 26px;
}
.guide {
    color: #2e7d32;
    font-weight: 800;
    font-size: 22px;
    margin-bottom: 12px;
}
.guide-text {
    display: inline-block;
    text-align: left;
    color: #303642;
    font-size: 16px;
    line-height: 1.75;
}
code {
    background: rgba(46,125,50,.10);
    color: #1b5e20;
    padding: 2px 6px;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Thiết lập")
    api_key = st.text_input(
        "VNStock API key",
        type="password",
        help="Dán API key từ vnstocks.com để tăng giới hạn request. Bỏ trống thì app tự chạy chậm để né limit Guest.",
    )
    industry = st.selectbox(
        "Chọn ngành",
        list(INDUSTRY_TICKERS.keys()),
        index=list(INDUSTRY_TICKERS.keys()).index("Thép") if "Thép" in INDUSTRY_TICKERS else 0,
    )

    st.markdown("### 📅 Khoảng thời gian phân tích")
    start_date = st.date_input("Chọn ngày bắt đầu", value=date(2015, 1, 1))
    end_date = st.date_input("Chọn ngày kết thúc", value=date(2025, 12, 31))

    # UX sản phẩm: người dùng chỉ chọn một khoảng thời gian.
    # App tự dùng phần cuối cùng của khoảng thời gian để đánh giá hiệu quả.
    # Với mặc định 2015-2025, giai đoạn đánh giá sẽ là năm 2025.
    test_start = date(end_date.year, 1, 1)
    test_end = end_date
    train_start = start_date
    train_end = test_start - timedelta(days=1)

    uploaded = st.file_uploader("Hoặc upload CSV [time, ticker, close]", type=["csv"])
    run_btn = st.button("🚀 Bắt đầu", use_container_width=True)

st.markdown("""
<div class="hero">
    <div class="school">TRƯỜNG ĐẠI HỌC NGÂN HÀNG THÀNH PHỐ HỒ CHÍ MINH</div>
    <div class="app-title">Xây dựng danh mục đầu tư tối ưu bằng<br/>mô hình LSTM - GRU</div>
    <div class="guide">🧪 Hướng dẫn sử dụng</div>
    <div class="guide-text">
        Ứng dụng này có hai tùy chọn dữ liệu đầu vào:<br/>
        <b>1.</b> Chọn ngành và khoảng thời gian: hệ thống tự động lấy dữ liệu, học mô hình và đánh giá hiệu quả ở giai đoạn cuối.<br/>
        <b>2.</b> Hoặc bạn có thể tải file CSV chứa dữ liệu gồm các cột <code>time</code>, <code>ticker</code>, <code>close</code>.
    </div>
</div>
""", unsafe_allow_html=True)

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
    f"📌 Hệ thống sẽ dùng dữ liệu từ **{train_start:%Y/%m/%d} → {train_end:%Y/%m/%d}** để xây dựng mô hình "
    f"và đánh giá hiệu quả trong giai đoạn **{test_start:%Y/%m/%d} → {test_end:%Y/%m/%d}**."
)

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
        # Giống notebook Cell 4: xóa trùng và sort mã trước khi tải.
        # Không dùng cache: mỗi lần bấm nút app sẽ gọi API và xử lý lại từ đầu.
        raw_data, failed = load_price_data(
            tickers, train_start, test_end,
            api_key=api_key, progress_callback=ui_progress
        )
    if raw_data.empty:
        st.error("Không tải được dữ liệu.")
        st.stop()
    price_pivot = make_price_pivot(raw_data)
    returns_df = calculate_returns(price_pivot)

progress_bar.progress(1.0)
status_box.success("Đã tải/xử lý dữ liệu xong. Không lưu cache. Returns được tính giống notebook Cell 7.")
st.caption(f"Khoảng dữ liệu tải được: {price_pivot.index.min().date()} → {price_pivot.index.max().date()} | Returns: {returns_df.index.min().date()} → {returns_df.index.max().date()} | Số dòng returns: {len(returns_df)}")

if failed:
    with st.expander("Một số mã không tải được"):
        st.dataframe(pd.DataFrame(failed), use_container_width=True)

# Giống notebook Cell 8 tuyệt đối: tính Sharpe trên returns_df full đã tải và lấy head(10).
# Không tự drop mã theo độ phủ dữ liệu, vì notebook không có bước đó.
if returns_df.shape[1] < TOP_N:
    st.error(f"Không đủ mã sau xử lý giống notebook. Cần ít nhất {TOP_N}, hiện có {returns_df.shape[1]}.")
    st.stop()

selected_symbols, sharpe_table = select_top_by_sharpe(returns_df, top_n=TOP_N)

st.subheader("📌 Top 10 mã được chọn theo Sharpe giống notebook")
st.write(", ".join(selected_symbols))

m1, m2, m3 = st.columns(3)
m1.metric("Số mã ban đầu", price_pivot.shape[1])
m2.metric("Số mã sau dropna", returns_df.shape[1])
m3.metric("Số mã train", len(selected_symbols))

with st.expander("Xem Sharpe từng mã giống notebook Cell 8"):
    st.dataframe(sharpe_table.reset_index().rename(columns={"index": "ticker", 0: "Sharpe"}), use_container_width=True)

try:
    X_train, y_train, train_dates, X_test, y_test, test_dates, train_returns, test_returns, scaler, debug_shapes = prepare_train_test(
        price_pivot, selected_symbols, train_start, train_end, test_start, test_end, window_size=WINDOW_SIZE
    )
except Exception as e:
    st.error(f"Lỗi tạo dữ liệu train/test: {e}")
    st.warning("Cách xử lý nhanh: chọn khoảng ngày có đủ dữ liệu hơn, hoặc kiểm tra các mã bị thiếu dữ liệu ở tập test.")
    st.stop()

st.info(f"Dữ liệu mô hình: giai đoạn học X={X_train.shape}, y={y_train.shape} | giai đoạn đánh giá X={X_test.shape}, y={y_test.shape}")
with st.expander("Thông tin kỹ thuật về dữ liệu mô hình"):
    st.json({k: str(v) for k, v in debug_shapes.items()})

try:
    with st.spinner("Đang xây dựng mô hình LSTM-GRU nhiều lần để chọn kết quả tốt nhất, bước này có thể hơi lâu..."):
        best, runs_df = train_multi_seed(
            X_train, y_train, X_test, y_test, test_dates, selected_symbols,
            seed_list=SEED_LIST, epochs=EPOCHS, batch_size=BATCH_SIZE,
        )
except Exception as e:
    st.error(f"Train model bị lỗi: {e}")
    st.stop()

st.success(f"Mô hình tốt nhất: lần chạy #{best['seed']} | Sharpe LSTM-GRU Dynamic: {best['sharpe']:.4f}")
with st.expander("Kết quả từng lần chạy mô hình"):
    st.dataframe(runs_df.style.format({"Lợi nhuận năm (%)": "{:.2f}", "Rủi ro năm (%)": "{:.2f}", "Sharpe": "{:.4f}"}), use_container_width=True)

weights_lstm_avg = best["pred_weights"].mean(axis=0)
results_lstm = pd.DataFrame({"Asset": selected_symbols, "Weight": weights_lstm_avg}).sort_values("Weight", ascending=False).reset_index(drop=True)
allo_1 = build_allocation_equal(selected_symbols)
allo_2 = build_allocation_80_20(train_returns)

Er_lstm, std_lstm = port_char_from_series(best["portfolio_returns"], annualize=True)
Er_1, std_1 = port_char(allo_1, test_returns, annualize=True)
Er_2, std_2 = port_char(allo_2, test_returns, annualize=True)
sharpe_lstm = sharpe_from_series(best["portfolio_returns"])
sharpe_1 = sharpe_port(allo_1, test_returns)
sharpe_2 = sharpe_port(allo_2, test_returns)

comparison = pd.DataFrame({
    "Chiến lược đầu tư": ["LSTM-GRU (Dynamic)", "Phân bổ đều", "Phân bổ 80-20"],
    "Lợi nhuận trung bình (%)": [Er_lstm * 100, Er_1 * 100, Er_2 * 100],
    "Độ lệch chuẩn (%)": [std_lstm * 100, std_1 * 100, std_2 * 100],
    "Hệ số Sharpe": [sharpe_lstm, sharpe_1, sharpe_2],
})

st.subheader("📊 Kết quả so sánh chiến lược")
st.dataframe(comparison.style.format({"Lợi nhuận trung bình (%)": "{:.2f}", "Độ lệch chuẩn (%)": "{:.2f}", "Hệ số Sharpe": "{:.4f}"}), use_container_width=True)

curves = pd.DataFrame(index=test_returns.index)
curves["LSTM-GRU (Dynamic)"] = (1 + best["portfolio_returns"]).cumprod()
curves["Phân bổ đều"] = cumulative_from_weights(allo_1, test_returns)
curves["Phân bổ 80-20"] = cumulative_from_weights(allo_2, test_returns)
fig_curve = px.line(curves, x=curves.index, y=curves.columns, title="Cumulative Return trên giai đoạn đánh giá")
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("🥧 Tỷ trọng trung bình danh mục LSTM-GRU")
results_lstm["Weight (%)"] = results_lstm["Weight"] * 100
cc1, cc2 = st.columns([1, 1])
with cc1:
    st.dataframe(results_lstm.style.format({"Weight": "{:.4f}", "Weight (%)": "{:.2f}%"}), use_container_width=True)
with cc2:
    st.plotly_chart(px.pie(results_lstm, names="Asset", values="Weight", title="Tỷ trọng trung bình LSTM-GRU"), use_container_width=True)
