import os, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RF_ANNUAL = 0.045
TRADING_DAYS = 252
LAMBDA_ENTROPY = 0.01
HORIZON = 5


def set_seed(seed=42):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.loc[common_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
    feat_list = []
    ret_1 = return_df.copy(); ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]; feat_list.append(ret_1)
    ret_5 = price_df.pct_change(5); ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]; feat_list.append(ret_5)
    ret_10 = price_df.pct_change(10); ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]; feat_list.append(ret_10)
    ma5 = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1; ma5.columns = [f"{c}_ma5_ratio" for c in ma5.columns]; feat_list.append(ma5)
    ma10 = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1; ma10.columns = [f"{c}_ma10_ratio" for c in ma10.columns]; feat_list.append(ma10)
    vol5 = return_df.rolling(5, min_periods=5).std(); vol5.columns = [f"{c}_vol5" for c in vol5.columns]; feat_list.append(vol5)
    vol10 = return_df.rolling(10, min_periods=10).std(); vol10.columns = [f"{c}_vol10" for c in vol10.columns]; feat_list.append(vol10)
    mom5 = price_df.pct_change(5); mom5.columns = [f"{c}_mom5" for c in mom5.columns]; feat_list.append(mom5)
    rsi14 = compute_rsi(price_df, 14) / 100.0; rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]; feat_list.append(rsi14)
    features = pd.concat(feat_list, axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return features


def create_sequences(features_scaled, target_returns, window_size=30, horizon=5):
    """Giống notebook Cell 12."""
    X, y, dates = [], [], []
    feat_values = features_scaled.values.astype(np.float32)
    target_values = target_returns.values.astype(np.float32)
    idx = features_scaled.index

    for i in range(len(features_scaled) - window_size - horizon + 1):
        X.append(feat_values[i:i + window_size])
        y.append(target_values[i + window_size:i + window_size + horizon].mean(axis=0))
        dates.append(idx[i + window_size + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)


def sharpe_loss_factory(rf_annual=RF_ANNUAL, trading_days=TRADING_DAYS, lambda_entropy=LAMBDA_ENTROPY):
    from tensorflow.keras import backend as K
    def sharpe_loss(y_true, y_pred):
        portfolio_returns = K.sum(y_true * y_pred, axis=1)
        portfolio_returns = portfolio_returns - (rf_annual / trading_days)
        mean_returns = K.mean(portfolio_returns)
        std_returns = K.std(portfolio_returns)
        sharpe = mean_returns / (std_returns + 1e-9)
        entropy = -K.sum(y_pred * K.log(y_pred + 1e-9), axis=1)
        entropy = K.mean(entropy)
        return -sharpe - lambda_entropy * entropy
    return sharpe_loss


def build_lstm_gru_model(timesteps, n_features, n_assets):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(96, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        GRU(48, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_assets, activation="softmax"),
    ])
    return model


def _slice_nonempty(df, start, end, name):
    out = df.loc[str(start):str(end)].copy()
    if out.empty:
        available = "không có dữ liệu" if df.empty else f"{df.index.min().date()} → {df.index.max().date()}"
        raise ValueError(f"{name} đang rỗng. Khoảng dữ liệu thực tế: {available}. Hãy chọn lại ngày.")
    return out


def prepare_train_test(price_pivot, selected_symbols, train_start, train_end, test_start, test_end, window_size=30):
    """
    Logic bám sát notebook:
    - Cắt train/test từ PRICE riêng
    - ffill/bfill giá riêng từng tập
    - pct_change để tạo train_returns/test_returns
    - build_features riêng cho train/test
    - fit scaler trên train, transform test
    """
    prices = price_pivot[selected_symbols].copy()

    train_prices = _slice_nonempty(prices, train_start, train_end, "Tập train prices").sort_index().ffill().bfill()
    test_prices = _slice_nonempty(prices, test_start, test_end, "Tập test prices").sort_index().ffill().bfill()

    train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    test_returns = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if train_returns.empty:
        raise ValueError("train_returns rỗng sau pct_change/dropna. Kiểm tra khoảng ngày train hoặc dữ liệu.")
    if test_returns.empty:
        raise ValueError("test_returns rỗng sau pct_change/dropna. Kiểm tra khoảng ngày test hoặc dữ liệu.")

    train_features = build_features(train_prices, train_returns)
    test_features = build_features(test_prices, test_returns)

    if train_features.empty:
        raise ValueError("train_features rỗng sau feature engineering. Cần train dài hơn hoặc tải lại dữ liệu.")
    if test_features.empty:
        raise ValueError("test_features rỗng sau feature engineering. Cần test dài hơn hoặc tải lại dữ liệu.")

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        index=train_features.index,
        columns=train_features.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        index=test_features.index,
        columns=test_features.columns,
    )

    train_target = train_returns.loc[train_scaled.index].copy()
    test_target = test_returns.loc[test_scaled.index].copy()

    X_train, y_train, train_dates = create_sequences(train_scaled, train_target, window_size=window_size, horizon=HORIZON)
    X_test, y_test, test_dates = create_sequences(test_scaled, test_target, window_size=window_size, horizon=HORIZON)

    if len(X_train) < 20 or len(X_test) < 5:
        raise ValueError(f"Dữ liệu train/test quá ít sau khi tạo sequence. X_train={len(X_train)}, X_test={len(X_test)}.")

    debug = {
        "train_prices": train_prices.shape,
        "test_prices": test_prices.shape,
        "train_returns": train_returns.shape,
        "test_returns": test_returns.shape,
        "train_features": train_features.shape,
        "test_features": test_features.shape,
    }

    return X_train, y_train, train_dates, X_test, y_test, test_dates, train_returns, test_returns, scaler, debug


def train_multi_seed(X_train, y_train, X_test, y_test, test_dates, symbols, seed_list=(7, 21, 42, 99, 123), epochs=100, batch_size=32):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    best = {"sharpe": -1e18}
    rows = []
    loss_fn = sharpe_loss_factory()
    for seed in seed_list:
        set_seed(seed)
        model = build_lstm_gru_model(X_train.shape[1], X_train.shape[2], y_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss_fn)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0, validation_split=0.2, callbacks=callbacks)
        pred = model.predict(X_test, verbose=0)
        weights_df = pd.DataFrame(pred, index=test_dates, columns=symbols)
        y_test_df = pd.DataFrame(y_test, index=test_dates, columns=symbols)
        port_ret = (weights_df * y_test_df).sum(axis=1)
        ann_ret = port_ret.mean() * TRADING_DAYS
        ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - RF_ANNUAL) / (ann_vol + 1e-12)
        row = {"seed": seed, "Lợi nhuận năm (%)": ann_ret * 100, "Rủi ro năm (%)": ann_vol * 100, "Sharpe": sharpe}
        rows.append(row)
        if sharpe > best["sharpe"]:
            best = {"seed": seed, "model": model, "history": history, "pred_weights": pred, "portfolio_returns": port_ret, "weights_df": weights_df, "sharpe": sharpe}
        tf.keras.backend.clear_session()
    return best, pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
