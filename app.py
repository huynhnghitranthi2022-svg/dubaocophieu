import random
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential

# ==========================================
# 1. CẤU HÌNH TRANG & CỐ ĐỊNH SEED (ỔN ĐỊNH KẾT QUẢ)
# ==========================================
st.set_page_config(
    page_title="Dự Báo Cổ Phiếu LSTM",
    page_icon="📈",
    layout="wide"
)

SEED = 42
def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds()

# ==========================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU & MÔ HÌNH (CÓ CACHE TĂNG TỐC)
# ==========================================
def format_ticker(ticker: str) -> str:
    """Tự động chuẩn hóa mã cổ phiếu (thêm .VN nếu là mã 3 ký tự chứng khoán VN)."""
    ticker = ticker.strip().upper()
    # Nếu người dùng nhập 3 chữ cái (VD: VNM, HPG, FPT, VIC) thì gợi ý/tự động dùng mã VN nếu cần
    return ticker

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Tải dữ liệu lịch sử từ Yahoo Finance và lưu cache trong 1 giờ."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty and len(ticker) == 3 and "." not in ticker:
        # Tự động thử thêm đuôi .VN nếu người dùng quên nhập
        df = yf.download(f"{ticker}.VN", start=start_date, end=end_date, progress=False)
    
    if not df.empty:
        # Xử lý MultiIndex trên các phiên bản yfinance mới
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].dropna()
    return df

def create_sequences(data: np.ndarray, window_size: int = 60):
    """Tạo chuỗi dữ liệu cửa sổ trượt (Sliding Window) cho LSTM."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def train_and_predict_lstm(
    ticker: str,
    start_date_str: str,
    end_date_str: str,
    window_size: int = 60,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Huấn luyện mô hình LSTM theo đúng kiến trúc chuẩn và dự báo 10 ngày tới.
    Được gắn @st.cache_resource để trả kết quả tức thì nếu tra cứu lại cùng mã.
    """
    set_seeds()
    df = load_stock_data(ticker, start_date_str, end_date_str)
    if df.empty or len(df) < window_size + 50:
        return None

    data = df[['Close']].values

    # 1. Chia tập Train (80%) và Test (20%)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    test_dates = df.index[train_size:]

    # 2. Chuẩn hóa dữ liệu bằng MinMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # 3. Tạo chuỗi dữ liệu (window_size = 60)
    X_train, y_train = create_sequences(train_scaled, window_size)
    test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
    X_test, y_test = create_sequences(test_combined, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 4. Chia tập Validation (10% cuối của tập Train)
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    # 5. Xây dựng mô hình LSTM
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 6. Huấn luyện mô hình
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # 7. Dự báo trên tập Test & Đánh giá
    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
    mae = mean_absolute_error(y_test_real, test_pred)
    r2 = r2_score(y_test_real, test_pred)

    # 8. Dự báo nhanh 10 ngày tiếp theo (Dùng Eager Call tối ưu tốc độ)
    last_60_days = np.vstack((train_scaled, test_scaled))[-window_size:]
    current_batch = last_60_days.reshape(1, window_size, 1).astype(np.float32)

    future_predictions = []
    for _ in range(10):
        # Gọi trực tiếp model(..., training=False) nhanh hơn ~15 lần so với model.predict() trong vòng lặp
        next_pred_tensor = model(current_batch, training=False)
        next_val = float(next_pred_tensor.numpy()[0, 0])
        future_predictions.append(next_val)
        
        next_reshaped = np.array([[[next_val]]], dtype=np.float32)
        current_batch = np.append(current_batch[:, 1:, :], next_reshaped, axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return {
        "df": df,
        "test_dates": test_dates,
        "y_test_real": y_test_real,
        "test_pred": test_pred,
        "future_predictions": future_predictions,
        "history": history.history,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "last_real_price": float(data[-1, 0]),
        "last_real_date": df.index[-1]
    }

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (SIDEBAR & MAIN UI)
# ==========================================
st.title("📈 Ứng Dụng Dự Báo Giá & Xu Hướng Cổ Phiếu (LSTM)")
st.markdown(
    "Nhập mã cổ phiếu để mô hình **Deep Learning LSTM** phân tích dữ liệu lịch sử "
    "và dự đoán xu hướng giá trong **10 ngày tiếp theo** tính từ ngày tra cứu."
)

with st.sidebar:
    st.header("⚙️ Thông Số Đầu Vào")
    ticker_raw = st.text_input(
        "Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, HPG.VN, AAPL):",
        value="VNM.VN"
    )
    
    # Ngày dự báo tính từ ngày nhập mã cổ phiếu
    input_date = st.date_input(
        "Ngày nhập mã (Mốc dự báo):",
        value=datetime.today()
    )
    
    start_date = st.date_input(
        "Lấy dữ liệu lịch sử từ ngày:",
        value=datetime(2018, 1, 1)
    )

    st.subheader("🔧 Tham số LSTM")
    epochs_input = st.slider("Số vòng lặp huấn luyện (Epochs):", min_value=5, max_value=30, value=10)
    batch_size_input = st.selectbox("Batch Size:", options=[16, 32, 64], index=1)
    
    run_btn = st.button("🚀 Phân Tích & Dự Báo Ngay", type="primary", use_container_width=True)

# ==========================================
# 4. HIỂN THỊ KẾT QUẢ & ĐỒ THỊ
# ==========================================
if run_btn or ticker_raw:
    ticker = format_ticker(ticker_raw)
    # Cộng thêm 1 ngày vào end_date vì yfinance lấy dữ liệu [start, end)
    end_date_str = (pd.to_datetime(input_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")

    t0 = time.time()
    with st.spinner(f"Đang tải dữ liệu và chạy mô hình LSTM cho mã **{ticker}**..."):
        result = train_and_predict_lstm(
            ticker=ticker,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            window_size=60,
            epochs=epochs_input,
            batch_size=batch_size_input
        )
    elapsed_time = time.time() - t0

    if result is None:
        st.error(
            f"❌ Không tìm thấy đủ dữ liệu cho mã **{ticker}**. "
            "Nếu là cổ phiếu Việt Nam, bạn hãy nhập kèm đuôi `.VN` (ví dụ: `VNM.VN`, `HPG.VN`, `VCB.VN`)."
        )
    else:
        y_test_real = result["y_test_real"]
        test_pred = result["test_pred"]
        future_preds = result["future_predictions"].flatten()
        metrics = result["metrics"]
        last_price = result["last_real_price"]
        
        # So sánh giá dự báo ngày thứ 10 với giá hiện tại và ngày đầu dự báo để xác định xu hướng
        day1_pred = future_preds[0]
        day10_pred = future_preds[-1]
        diff_vs_current = day10_pred - last_price
        pct_vs_current = (diff_vs_current / last_price) * 100
        
        diff_10days_internal = day10_pred - day1_pred
        is_uptrend = diff_10days_internal >= 0

        # --- KHU VỰC 1: THÔNG BÁO XU HƯỚNG & METRICS ---
        if is_uptrend:
            st.success(
                f"📈 **XU HƯỚNG MÔ HÌNH TRONG 10 NGÀY TỚI: TĂNG TRƯỞNG** | "
                f"Dự báo từ **{day1_pred:,.0f}** (Ngày 1) lên **{day10_pred:,.0f}** (Ngày 10) "
                f"— Chênh lệch so với giá đóng cửa gần nhất ({last_price:,.0f}): **{pct_vs_current:+.2f}%**"
            )
        else:
            st.warning(
                f"📉 **XU HƯỚNG MÔ HÌNH TRONG 10 NGÀY TỚI: GIẢM / ĐIỀU CHỈNH** | "
                f"Dự báo từ **{day1_pred:,.0f}** (Ngày 1) xuống **{day10_pred:,.0f}** (Ngày 10) "
                f"— Chênh lệch so với giá đóng cửa gần nhất ({last_price:,.0f}): **{pct_vs_current:+.2f}%**"
            )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Giá Đóng Cửa Gần Nhất", f"{last_price:,.0f}")
        col2.metric("Dự Báo Ngày Thứ 10", f"{day10_pred:,.0f}", f"{pct_vs_current:+.2f}%")
        col3.metric("Sai số RMSE", f"{metrics['rmse']:,.2f}")
        col4.metric("Sai số MAE", f"{metrics['mae']:,.2f}")
        col5.metric("Độ phù hợp R²", f"{metrics['r2']:.4f}", f"⚡ {elapsed_time:.2f}s")

        st.divider()

        # --- KHU VỰC 2: CÁC ĐỒ THỊ BIỂU DIỄN ---
        tab1, tab2, tab3 = st.tabs([
            "📊 Đồ Thị Thực Tế vs Dự Báo (Giống Ảnh Mẫu)",
            "🚀 Đồ Thị Dự Báo Xu Hướng 10 Ngày Tới",
            "📉 Biểu Đồ Loss (Train vs Validation)"
        ])

        with tab1:
            st.subheader(f"So sánh Giá thật và Dự báo của mô hình LSTM ({ticker})")
            # Vẽ đúng chuẩn giao diện Matplotlib trong ảnh mẫu của bạn
            fig1, ax1 = plt.subplots(figsize=(12, 5.5))
            ax1.plot(y_test_real, label="Giá thật", color="#1f77b4", linewidth=1.5)
            ax1.plot(test_pred, label="Dự báo", color="#ff7f0e", linewidth=1.8)
            ax1.legend(loc="upper right", fontsize=11)
            ax1.set_ylabel("Giá cổ phiếu")
            ax1.set_xlabel("Số phiên giao dịch trên tập Test")
            ax1.grid(False)  # Giữ nền trắng trơn giống hệt ảnh bạn cung cấp
            st.pyplot(fig1)

        with tab2:
            col_chart_all, col_chart_zoom = st.columns([3, 2])
            
            with col_chart_all:
                st.subheader("Toàn cảnh Tập Test & Dự báo 10 ngày tới")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(y_test_real, label="Giá thật", color="#1f77b4", linewidth=1.5)
                ax2.plot(test_pred, label="Dự báo test", color="#ff7f0e", linewidth=1.8)
                future_x = range(len(y_test_real), len(y_test_real) + 10)
                trend_color = "#2ca02c" if is_uptrend else "#d62728"
                ax2.plot(
                    future_x,
                    future_preds,
                    marker="o",
                    color=trend_color,
                    linewidth=2,
                    label="10 ngày tới"
                )
                ax2.legend(loc="upper right")
                ax2.grid(True, linestyle="--", alpha=0.3)
                st.pyplot(fig2)

            with col_chart_zoom:
                st.subheader("Cận cảnh xu hướng 10 ngày tới")
                fig_zoom, ax_zoom = plt.subplots(figsize=(7, 5))
                # Hiển thị 30 phiên gần nhất + 10 ngày dự báo để nhìn rõ xu hướng
                recent_len = 30
                recent_real = y_test_real[-recent_len:].flatten()
                recent_pred = test_pred[-recent_len:].flatten()
                x_recent = np.arange(recent_len)
                x_future = np.arange(recent_len, recent_len + 10)

                ax_zoom.plot(x_recent, recent_real, label="Giá thật (30 phiên cuối)", color="#1f77b4")
                ax_zoom.plot(x_recent, recent_pred, label="Dự báo (30 phiên cuối)", color="#ff7f0e")
                ax_zoom.plot(
                    np.insert(x_future, 0, x_recent[-1]),
                    np.insert(future_preds, 0, recent_pred[-1]),
                    marker="o",
                    color=trend_color,
                    linewidth=2.2,
                    label="Xu hướng 10 ngày tới"
                )
                ax_zoom.axvspan(x_recent[-1], x_future[-1], color=trend_color, alpha=0.1)
                ax_zoom.legend(loc="best")
                ax_zoom.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig_zoom)

        with tab3:
            st.subheader("Quá trình hội tụ hàm mất mát (Loss Train vs Validation)")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(result["history"]["loss"], marker="o", label="Train Loss")
            ax3.plot(result["history"]["val_loss"], marker="s", label="Validation Loss")
            ax3.set_title("Loss Train vs Validation")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Mean Squared Error (MSE)")
            ax3.legend()
            ax3.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig3)

        # --- KHU VỰC 3: BẢNG DỮ LIỆU DỰ BÁO CHI TIẾT ---
        st.subheader(f"📋 Bảng Dự Báo Chi Tiết 10 Ngày Tới (Tính từ {pd.to_datetime(input_date).strftime('%d/%m/%Y')})")
        
        future_dates = [pd.to_datetime(input_date) + timedelta(days=i) for i in range(1, 11)]
        table_rows = []
        prev_val = last_price
        for idx, (f_date, f_val) in enumerate(zip(future_dates, future_preds), start=1):
            step_change = f_val - prev_val
            step_pct = (step_change / prev_val) * 100
            trend_icon = "🟢 Tăng" if step_change >= 0 else "🔴 Giảm"
            table_rows.append({
                "Ngày thứ": f"Ngày +{idx}",
                "Ngày dự kiến": f_date.strftime("%d/%m/%Y"),
                "Giá dự báo": f"{f_val:,.2f}",
                "Thay đổi so với ngày trước": f"{step_change:+,.2f} ({step_pct:+.2f}%)",
                "Trạng thái": trend_icon
            })
            prev_val = f_val

        forecast_table = pd.DataFrame(table_rows)
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
