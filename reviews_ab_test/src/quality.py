"""الخطوة 3: تجهيز البيانات وفحوص الجودة قبل أي تحليل."""
import numpy as np
import pandas as pd
from scipy.stats import chisquare

REQUIRED = ["user_id", "variant", "day", "pre_orders_90d", "ordered", "revenue",
            "profit", "contacted_support", "page_load_ms"]


def clean(raw):
    """نتحقق من الأعمدة والقيم، ونحذف السجلات المكررة لنفس المستخدم."""
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"أعمدة ناقصة: {missing_cols}")
    report = {
        "rows_raw": len(raw),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_user_ids": int(raw["user_id"].duplicated().sum()),
        "missing_required": int(raw[REQUIRED].isna().sum().sum()),
        "bad_variant_values": int((~raw["variant"].isin(["A", "B"])).sum()),
    }
    df = raw.drop_duplicates(subset="user_id").reset_index(drop=True)
    # order_value وcancelled فارغة منطقيًا لمن لم يطلب، فلا تعتبر بيانات مفقودة
    consistent = ((df["ordered"] == 1) == df["order_value"].notna()).all()
    report["rows_clean"] = len(df)
    report["order_fields_consistent"] = bool(consistent)
    return df, report


def srm(df, expected_share_b, threshold=0.001):
    """اختبار مربع كاي لاختلال نسبة التوزيع (Sample Ratio Mismatch)."""
    n_a = int((df["variant"] == "A").sum())
    n_b = int((df["variant"] == "B").sum())
    total = n_a + n_b
    chi2, p = chisquare([n_a, n_b], [total * (1 - expected_share_b), total * expected_share_b])
    return {"n_A": n_a, "n_B": n_b, "share_B": n_b / total, "chi2": chi2, "p": p, "ok": p >= threshold}


def balance(df):
    """التوازن في خصائص ما قبل التجربة: الفرق المعياري (SMD) أقل من 0.1 يعني توازنًا جيدًا."""
    rows = []
    covs = {"pre_orders_90d": df["pre_orders_90d"], "returning": df["returning"],
            "mobile": (df["device"] == "mobile").astype(int)}
    for name, x in covs.items():
        a, b = x[df["variant"] == "A"], x[df["variant"] == "B"]
        smd = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rows.append({"covariate": name, "mean_A": a.mean(), "mean_B": b.mean(), "SMD": smd, "balanced": abs(smd) < 0.1})
    return pd.DataFrame(rows)
