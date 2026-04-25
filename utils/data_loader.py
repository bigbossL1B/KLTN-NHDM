import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from vnstock import Vnstock


def _apply_api_key(api_key: str | None):
    """Set common env names used by vnstock/vnstock_data builds.
    Different vnstock versions read different names, so setting all is harmless.
    """
    if api_key:
        key = str(api_key).strip()
        for name in ["VNSTOCK_API_KEY", "VNS_API_KEY", "VNSTOCK_TOKEN", "VNS_TOKEN", "API_KEY"]:
            os.environ[name] = key


def _parse_wait_seconds(error_text: str, default_wait: int = 45) -> int:
    text = str(error_text)
    m = re.search(r"Chờ\s+(\d+)\s+giây|Wait\s+to\s+retry\D*(\d+)", text, flags=re.I)
    if not m:
        m = re.search(r"wait\D*(\d+)\D*second", text, flags=re.I)
    if m:
        nums = [g for g in m.groups() if g]
        if nums:
            return max(int(nums[0]) + 3, 5)
    return default_wait


def load_price_data(
    tickers,
    start_date,
    end_date,
    source="KBS",
    sleep_sec=4.0,
    max_retry=5,
    max_per_minute=15,
    api_key=None,
    progress_callback=None,
):
    """Load OHLCV data from vnstock with rate-limit pacing and UI callbacks."""
    _apply_api_key(api_key)

    all_data, failed = [], []
    request_count = 0
    window_start = time.time()
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        ok, last_err = False, None
        if progress_callback:
            progress_callback(idx, total, ticker, "loading", None)

        for attempt in range(1, max_retry + 1):
            try:
                stock = Vnstock().stock(symbol=ticker, source=source)
                df = stock.quote.history(start=str(start_date), end=str(end_date), interval="1D")

                if df is None or df.empty:
                    last_err = "empty data"
                    break

                df = df.copy()
                if "time" not in df.columns:
                    if "date" in df.columns:
                        df["time"] = df["date"]
                    elif "datetime" in df.columns:
                        df["time"] = df["datetime"]
                    else:
                        raise ValueError("Missing time/date/datetime column")

                df["ticker"] = ticker
                keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
                all_data.append(df[keep_cols])
                ok = True
                request_count += 1

                if progress_callback:
                    progress_callback(idx, total, ticker, "ok", f"{len(df)} dòng")
                break

            except Exception as e:
                last_err = str(e)
                msg = last_err.lower()
                if "rate limit" in msg or "429" in msg or "too many" in msg:
                    wait_s = _parse_wait_seconds(last_err, default_wait=45)
                    if progress_callback:
                        progress_callback(idx, total, ticker, "rate_limit", f"Chờ {wait_s}s rồi thử lại lần {attempt}/{max_retry}")
                    time.sleep(wait_s)
                else:
                    if progress_callback:
                        progress_callback(idx, total, ticker, "retry", f"Lỗi, thử lại lần {attempt}/{max_retry}: {e}")
                    time.sleep(2)

        if not ok:
            failed.append({"ticker": ticker, "error": last_err})
            if progress_callback:
                progress_callback(idx, total, ticker, "failed", last_err)

        # Chủ động nghỉ để tài khoản Guest không vượt 20 requests/phút.
        time.sleep(sleep_sec)
        if request_count >= max_per_minute:
            elapsed = time.time() - window_start
            if elapsed < 60:
                wait_s = int(60 - elapsed) + 2
                if progress_callback:
                    progress_callback(idx, total, ticker, "cooldown", f"Nghỉ {wait_s}s để né rate limit")
                time.sleep(wait_s)
            request_count = 0
            window_start = time.time()

    if not all_data:
        return pd.DataFrame(), failed
    return pd.concat(all_data, ignore_index=True), failed


def load_or_download_price_data(
    tickers,
    start_date,
    end_date,
    cache_key,
    api_key=None,
    force_reload=False,
    progress_callback=None,
):
    """Use local CSV cache first so Streamlit rerun không gọi API lại."""
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.csv"

    if cache_file.exists() and not force_reload:
        raw = pd.read_csv(cache_file)
        if progress_callback:
            progress_callback(len(tickers), len(tickers), "CACHE", "cache", f"Đã load cache: {cache_file}")
        return raw, []

    # Guest nên <=15/min; có API key thì tăng nhẹ nhưng vẫn pacing để tránh lỗi.
    max_per_minute = 50 if api_key else 15
    sleep_sec = 1.4 if api_key else 4.0
    raw, failed = load_price_data(
        tickers,
        start_date,
        end_date,
        api_key=api_key,
        sleep_sec=sleep_sec,
        max_per_minute=max_per_minute,
        progress_callback=progress_callback,
    )
    if not raw.empty:
        raw.to_csv(cache_file, index=False)
    return raw, failed


def make_price_pivot(raw_data):
    df = raw_data.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    pivot = df.pivot_table(index="time", columns="ticker", values="close", aggfunc="last")
    return pivot.sort_index()


def calculate_returns(price_pivot):
    """Giống notebook Cell 7 tuyệt đối:
    - Fill missing trên GIÁ trước
    - pct_change
    - replace inf/nan
    - drop dòng có NaN bất kỳ

    Lưu ý: KHÔNG drop cột theo ngưỡng 70%, vì notebook không làm bước đó.
    Nếu drop cột trước, tập ngày dùng để tính Sharpe sẽ đổi và Top 10 sẽ lệch.
    """
    price_filled = price_pivot.replace([np.inf, -np.inf], np.nan).ffill()
    returns_df = price_filled.pct_change()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.dropna(how="any")
    return returns_df
