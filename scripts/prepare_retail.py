"""Convert UCI Online Retail II (CC BY 4.0) from Excel to Parquet.

    python scripts/prepare_retail.py online_retail_II.xlsx out_dir
"""
import sys
from pathlib import Path

import pandas as pd


def main(xlsx: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]
    df["customer_id"] = df["customer_id"].astype("Int64")
    df.to_parquet(out / "online_retail_ii.parquet", compression="zstd", index=False)
    print("rows", len(df), "customers", df["customer_id"].nunique(), "items", df["stock_code"].nunique(),
          "dates", df["invoice_date"].min(), df["invoice_date"].max())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
