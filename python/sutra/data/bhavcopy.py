"""NSE daily bhavcopy downloader and panel builder.

Why bhavcopies: each day's file lists every stock that traded that day,
so a universe built from them includes names that later delisted (no
survivorship bias), with no paid data vendor.

Two archive formats (check the URLs before a big download; NSE changes them):
  legacy (to early Jul 2024): archives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip
  UDiFF  (from Jul 2024):     nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip
NSE rejects non-browser clients, so send a browser User-Agent and go slowly.
"""
import datetime as dt
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
UDIFF_START = dt.date(2024, 7, 8)

# normalised column names -> (legacy name, UDiFF name)
COLUMNS = {
    "symbol": ("SYMBOL", "TckrSymb"),
    "series": ("SERIES", "SctySrs"),
    "isin": ("ISIN", "ISIN"),
    "open": ("OPEN", "OpnPric"),
    "high": ("HIGH", "HghPric"),
    "low": ("LOW", "LwPric"),
    "close": ("CLOSE", "ClsPric"),
    "prev_close": ("PREVCLOSE", "PrvsClsgPric"),
    "volume": ("TOTTRDQTY", "TtlTradgVol"),
    "value": ("TOTTRDVAL", "TtlTrfVal"),
}


def url_for(day):
    if day >= UDIFF_START:
        return ("https://nsearchives.nseindia.com/content/cm/"
                f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
    mon = day.strftime("%b").upper()
    return ("https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{day:%Y}/{mon}/cm{day:%d}{mon}{day:%Y}bhav.csv.zip")


def download(start, end, out_dir, pause=0.7):
    """Save raw zip files to out_dir; skips days already on disk and holidays (404)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = start
    while day <= end:
        f = out / f"{day:%Y%m%d}.csv.zip"
        if day.weekday() < 5 and not f.exists():
            try:
                req = urllib.request.Request(url_for(day), headers=UA)
                f.write_bytes(urllib.request.urlopen(req, timeout=30).read())
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            time.sleep(pause)
        day += dt.timedelta(days=1)


def read_day(path):
    """One raw zip -> DataFrame with normalised columns, EQ series only."""
    import pandas as pd

    day = dt.datetime.strptime(Path(path).name[:8], "%Y%m%d").date()
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df.columns = df.columns.str.strip()
    idx = 1 if day >= UDIFF_START else 0
    df = df.rename(columns={v[idx]: k for k, v in COLUMNS.items()})[list(COLUMNS)]
    df = df[df["series"].str.strip() == "EQ"]
    df["date"] = day
    return df


def adjustment_factor(close_yesterday, prev_close_today):
    """NSE restates PREVCLOSE on the ex-date of a split/bonus, so the ratio
    prev_close_today / close_yesterday is the corporate-action factor for that
    day (1.0 on normal days). Key the panel on ISIN, not symbol: symbols change."""
    return prev_close_today / close_yesterday
