"""train_all.py — Train every pillar, save models, print + save the honest scorecard."""
import json, pandas as pd
import config as C
import pillar1_market_research as p1
import pillar2_demand as p2
import pillar3_segmentation as p3
import pillar4_pmf as p4

if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH)
    print("Loaded", df.shape)

    print("\n[1/4] Market Research ...")
    m1 = p1.train(df)
    print("    ", m1)

    print("[2/4] Demand Forecast ...")
    m2 = p2.train(df)
    print("    ", m2)

    print("[3/4] Customer Segmentation (retrieval eval) ...")
    m3 = p3.evaluate(df)
    print("    ", m3)

    print("[4/4] Product-Market Fit / Failure ...")
    m4 = p4.train(df)
    print("    ", m4)

    report = {
        "market_research": m1,
        "demand_forecast": m2,
        "customer_segmentation": m3,
        "product_market_fit": m4,
    }
    from pathlib import Path
    out = Path(__file__).parent.parent / "outputs" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")
