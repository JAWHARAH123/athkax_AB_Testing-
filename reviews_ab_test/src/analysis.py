"""الخطوة 4: التحليل الإحصائي.

- المقياس الرئيسي: اختبار z لنسبتين + فترة ثقة، ثم تعديل بالانحدار باستخدام بيانات ما قبل التجربة
  (فكرة CUPED نفسها: نستخدم متغيرات لا تتأثر بالمعالجة لتقليل التباين).
- المقاييس الثانوية: اختبار Welch + تصحيح Benjamini-Hochberg، وBootstrap للإيراد لأنه ملتوٍ.
- مقاييس الحماية: اختبار عدم الإضرار بحدود مسموحة محددة مسبقًا.
- فحص أثر الحداثة، وتحليل استكشافي حسب الجهاز.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

Z95 = stats.norm.ppf(0.975)


def mean_diff(a, b):
    """فرق المتوسطات (B − A) مع فترة ثقة 95% بخطأ معياري Welch."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p = 2 * stats.norm.sf(abs(diff / se))
    return {"mean_A": a.mean(), "mean_B": b.mean(), "diff": diff,
            "ci_low": diff - Z95 * se, "ci_high": diff + Z95 * se, "se": se, "p": p}


def primary(df):
    A, B = df[df.variant == "A"], df[df.variant == "B"]
    x = np.array([B["ordered"].sum(), A["ordered"].sum()])
    n = np.array([len(B), len(A)])
    z, p = proportions_ztest(x, n)
    res = mean_diff(A["ordered"], B["ordered"])
    res.update({"z": z, "p": p, "relative_lift": res["diff"] / res["mean_A"]})

    # تعديل بالانحدار (ANCOVA/CUPED): نفس الأثر، لكن بتباين أقل
    d = df.assign(treat=(df.variant == "B").astype(int), mobile=(df.device == "mobile").astype(int))
    model = smf.ols("ordered ~ treat + pre_orders_90d + returning + mobile", data=d).fit(cov_type="HC1")
    lo, hi = model.conf_int().loc["treat"]
    adjusted = {"diff": model.params["treat"], "ci_low": lo, "ci_high": hi,
                "se": model.bse["treat"], "p": model.pvalues["treat"],
                "variance_reduction": 1 - (model.bse["treat"] / res["se"]) ** 2}
    return res, adjusted


def bootstrap_diff(a, b, n_boot=2000, seed=7):
    """فترة ثقة بالـ Bootstrap لفرق المتوسطات (مناسبة للمقاييس الملتوية مثل الإيراد)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
    return np.percentile(diffs, [2.5, 97.5])


def secondary(df):
    A, B = df[df.variant == "A"], df[df.variant == "B"]
    metrics = {"revenue": "الإيراد لكل مستخدم (ريال)", "order_value": "قيمة الطلب لمن طلب (ريال)"}
    rows = []
    for col, label in metrics.items():
        r = mean_diff(A[col], B[col])
        r["metric"] = label
        rows.append(r)
    out = pd.DataFrame(rows).set_index("metric")
    out["significant_BH"] = multipletests(out["p"], alpha=0.05, method="fdr_bh")[0]
    lo, hi = bootstrap_diff(A["revenue"], B["revenue"])
    out.loc["الإيراد لكل مستخدم (ريال)", "boot_ci_low"] = lo
    out.loc["الإيراد لكل مستخدم (ريال)", "boot_ci_high"] = hi
    return out


def guardrails(df, defs):
    """المقياس يفشل إذا تجاوزت فترة الثقة الحد المسموح في الاتجاه السيئ."""
    A, B = df[df.variant == "A"], df[df.variant == "B"]
    rows = []
    for col, g in defs.items():
        r = mean_diff(A[col], B[col])
        if g["bad"] == "up":
            ok = r["ci_high"] <= g["margin"]
        else:
            ok = r["ci_low"] >= -g["margin"]
        rows.append({"metric": g["label"], **r, "margin": g["margin"], "bad_direction": g["bad"],
                     "status": "سليم" if ok else "تجاوز الحد"})
    return pd.DataFrame(rows).set_index("metric")


def novelty(df):
    """الأثر لكل يوم دخول، ومقارنة الأسبوع الأول بالثاني."""
    rows = []
    for d, g in df.groupby("day"):
        r = mean_diff(g.loc[g.variant == "A", "ordered"], g.loc[g.variant == "B", "ordered"])
        rows.append({"day": d, **r})
    daily = pd.DataFrame(rows)
    w1 = df[df.day <= 7]; w2 = df[df.day > 7]
    week = {
        "week1": mean_diff(w1.loc[w1.variant == "A", "ordered"], w1.loc[w1.variant == "B", "ordered"]),
        "week2": mean_diff(w2.loc[w2.variant == "A", "ordered"], w2.loc[w2.variant == "B", "ordered"]),
    }
    d = df.assign(treat=(df.variant == "B").astype(int), week2=(df.day > 7).astype(int))
    m = smf.ols("ordered ~ treat * week2", data=d).fit(cov_type="HC1")
    week["interaction_p"] = m.pvalues["treat:week2"]
    return daily, week


def by_segment(df, col="device"):
    """تحليل استكشافي فقط (غير مخطط كقرار)، فلا نبني عليه قرار الإطلاق."""
    rows = []
    for seg, g in df.groupby(col):
        r = mean_diff(g.loc[g.variant == "A", "ordered"], g.loc[g.variant == "B", "ordered"])
        rows.append({col: seg, "n": len(g), **r})
    return pd.DataFrame(rows)


def aa_simulation(df, runs=500, seed=11):
    """اختبارات A/A: نقسم مستخدمي A عشوائيًا إلى نصفين ونختبر. نتوقع حوالي 5% نتائج دالة بالصدفة."""
    rng = np.random.default_rng(seed)
    y = df.loc[df.variant == "A", "ordered"].to_numpy()
    pvals = np.empty(runs)
    for i in range(runs):
        mask = rng.random(len(y)) < 0.5
        pvals[i] = mean_diff(y[mask], y[~mask])["p"]
    return pvals


def decide(ci_low, ci_high, mde, guardrails_ok):
    if ci_high < 0:
        return "لا نطلق", "النسخة B خفضت نسبة الشراء."
    if not guardrails_ok:
        return "لا نطلق الآن", "أحد مقاييس الحماية تجاوز الحد المسموح؛ نعالجه ثم نعيد الاختبار."
    if ci_low > 0:
        return "نطلق", "الأثر إيجابي بثقة 95% وكل مقاييس الحماية ضمن الحدود."
    if ci_high < mde:
        return "لا نطلق", "استبعدنا وجود أثر بالحجم الذي يهمنا تجاريًا."
    return "النتيجة غير حاسمة", "نمدد التجربة أو نعيد تصميمها بقوة أعلى."
