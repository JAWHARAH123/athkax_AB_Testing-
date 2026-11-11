"""
مشروع: هل عرض تقييمات المستخدمين بجانب المنتجات يزيد معدل الشراء؟
برنامج التجارب واختبارات A/B والاستدلال السببي - أكاديمية سدايا

التشغيل:
    pip install -r requirements.txt
    python main.py                # يستخدم data/dataset.csv إن وجد، وإلا ينشئه
    python main.py --regenerate   # يعيد إنشاء البيانات بالمحاكاة

الخطوات:
    1) حجم العينة والمدة   2) إنشاء البيانات   3) فحوص الجودة
    4) التحليل             5) الرسوم            6) التقرير والتوصية
"""
import argparse
import json

import pandas as pd

from src import analysis, design, plots, quality, simulate
from src.config import CHARTS_DIR, DATA_PATH, DESIGN, GUARDRAILS, RESULTS_DIR, SIMULATION
from src.utils import header, pct, pts


def main(regenerate=False):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    header("1) حجم العينة والمدة")
    pl = design.plan(DESIGN)
    print(f"خط الأساس {pct(DESIGN['baseline_rate'])}، وأصغر أثر يهمنا {pts(DESIGN['mde_abs'])}، "
          f"α={DESIGN['alpha']}، والقوة {DESIGN['power']:.0%}")
    print(f"حجم العينة لكل مجموعة: {pl['n_per_group_formula']:,} (الصيغة) / {pl['n_per_group_statsmodels']:,} (statsmodels)")
    print(f"نحتاج {pl['raw_days']} يومًا، وقربناها إلى {pl['days']} يومًا (أسابيع كاملة)")
    print(f"المتوقع خلال المدة: {pl['n_expected_per_group']:,} لكل مجموعة، "
          f"وأصغر أثر قابل للاكتشاف {pl['mde_planned'] * 100:.2f} نقطة "
          f"(لو اكتفينا بأسبوع: {pl['mde_one_week'] * 100:.2f} نقطة)")

    # ------------------------------------------------------------------ 2
    header("2) البيانات")
    if regenerate or not DATA_PATH.exists():
        raw = simulate.generate(DESIGN, SIMULATION, pl["days"])
        simulate.save(raw, DATA_PATH)
        print(f"أُنشئت البيانات بالمحاكاة وحُفظت في {DATA_PATH.relative_to(DATA_PATH.parents[1])}")
    raw = pd.read_csv(DATA_PATH)
    print(f"عدد السجلات الخام: {len(raw):,}، والأعمدة: {', '.join(raw.columns)}")

    # ------------------------------------------------------------------ 3
    header("3) فحوص الجودة")
    df, q = quality.clean(raw)
    print(f"سجلات مكررة حُذفت: {q['duplicate_user_ids']:,}، والمتبقي {q['rows_clean']:,} مستخدم")
    print(f"قيم مفقودة في الأعمدة الأساسية: {q['missing_required']}، وقيم نسخة غير صحيحة: {q['bad_variant_values']}")
    s = quality.srm(df, DESIGN["treatment_share"])
    print(f"SRM: A={s['n_A']:,}  B={s['n_B']:,}  نسبة B={s['share_B']:.2%}  χ²={s['chi2']:.2f}  p={s['p']:.3f}  "
          f"→ {'التوزيع سليم' if s['ok'] else 'خلل في التوزيع! لا تثق بالنتائج'}")
    bal = quality.balance(df)
    print("التوازن قبل التجربة (SMD):", ", ".join(f"{r.covariate}={r.SMD:+.3f}" for r in bal.itertuples()))
    if not s["ok"]:
        print("توقف التحليل بسبب SRM. ابحث عن السبب أولًا.")
        return

    # ------------------------------------------------------------------ 4
    header("4) التحليل")
    prim, adj = analysis.primary(df)
    print(f"نسبة الشراء: A={pct(prim['mean_A'])}  B={pct(prim['mean_B'])}")
    print(f"الأثر: {pts(prim['diff'])} (فترة الثقة 95%: {prim['ci_low'] * 100:.2f} إلى {prim['ci_high'] * 100:.2f})، "
          f"p={prim['p']:.2g}، زيادة نسبية {prim['relative_lift']:.1%}")
    print(f"مع التعديل بالانحدار: {pts(adj['diff'])} ({adj['ci_low'] * 100:.2f} إلى {adj['ci_high'] * 100:.2f})، "
          f"وانخفاض التباين {adj['variance_reduction']:.1%}")

    sec = analysis.secondary(df)
    print("\nالمقاييس الثانوية (بعد تصحيح BH):")
    for name, r in sec.iterrows():
        print(f"  {name}: الفرق {r['diff']:+.2f} ({r['ci_low']:.2f} إلى {r['ci_high']:.2f})، p={r['p']:.3g}، "
              f"دال={'نعم' if r['significant_BH'] else 'لا'}")

    guard = analysis.guardrails(df, GUARDRAILS)
    print("\nمقاييس الحماية:")
    for name, r in guard.iterrows():
        print(f"  {name}: الفرق {r['diff']:+.4f} ({r['ci_low']:.4f} إلى {r['ci_high']:.4f})، الحد {r['margin']} → {r['status']}")
    guard_ok = (guard["status"] == "سليم").all()

    daily, week = analysis.novelty(df)
    print(f"\nأثر الحداثة: الأسبوع الأول {pts(week['week1']['diff'])}، والثاني {pts(week['week2']['diff'])}، "
          f"p للفرق بينهما={week['interaction_p']:.2f}")
    seg = analysis.by_segment(df)
    print("حسب الجهاز (استكشافي):", ", ".join(f"{r.device} {pts(r.diff)}" for r in seg.itertuples()))
    aa = analysis.aa_simulation(df)
    print(f"اختبارات A/A ({len(aa)}): {(aa < 0.05).mean():.1%} دالة بالصدفة (المتوقع حوالي 5%)")

    # ------------------------------------------------------------------ 5
    header("5) الرسوم")
    plots.conversion_bars(prim, CHARTS_DIR / "01_conversion_by_variant.png")
    plots.effect_forest(prim, adj, DESIGN["mde_abs"], CHARTS_DIR / "02_effect_with_ci.png")
    plots.daily_effect(daily, prim["diff"], CHARTS_DIR / "03_novelty_daily_effect.png")
    plots.guardrail_chart(guard, CHARTS_DIR / "04_guardrails.png")
    plots.aa_hist(aa, CHARTS_DIR / "05_aa_tests.png")
    print(f"حُفظت 5 رسوم في outputs/charts")

    # ------------------------------------------------------------------ 6
    header("6) التقرير والتوصية")
    decision, reason = analysis.decide(prim["ci_low"], prim["ci_high"], DESIGN["mde_abs"], guard_ok)
    profit = guard.loc[GUARDRAILS["profit"]["label"]]
    annual = {k: profit[k] * DESIGN["annual_users"] / 1e6 for k in ["diff", "ci_low", "ci_high"]}
    print(f"القرار: {decision}. {reason}")
    print(f"التغير التقديري في الربح السنوي: {annual['diff']:.2f} مليون ريال "
          f"({annual['ci_low']:.2f} إلى {annual['ci_high']:.2f})")

    # حفظ النتائج
    pd.DataFrame([{"metric": "نسبة الشراء (بدون تعديل)", **prim},
                  {"metric": "نسبة الشراء (تعديل بالانحدار)", **adj}]).to_csv(
        RESULTS_DIR / "primary_metric.csv", index=False, encoding="utf-8-sig")
    sec.to_csv(RESULTS_DIR / "secondary_metrics.csv", encoding="utf-8-sig")
    guard.to_csv(RESULTS_DIR / "guardrails.csv", encoding="utf-8-sig")
    daily.to_csv(RESULTS_DIR / "daily_effect.csv", index=False, encoding="utf-8-sig")
    seg.to_csv(RESULTS_DIR / "segment_device.csv", index=False, encoding="utf-8-sig")
    bal.to_csv(RESULTS_DIR / "balance_check.csv", index=False, encoding="utf-8-sig")
    summary = {"plan": pl, "quality": q, "srm": s, "decision": decision, "reason": reason,
               "primary": prim, "adjusted": adj, "annual_profit_million_sar": annual,
               "novelty": {"week1": week["week1"]["diff"], "week2": week["week2"]["diff"],
                           "interaction_p": week["interaction_p"]},
               "aa_false_positive_rate": float((aa < 0.05).mean())}
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    write_report(pl, s, prim, adj, sec, guard, week, annual, decision, reason)
    print("حُفظت النتائج والتقرير في outputs/results")


def write_report(pl, s, prim, adj, sec, guard, week, annual, decision, reason):
    rev = sec.loc["الإيراد لكل مستخدم (ريال)"]
    guard_rows = "\n".join(
        f"| {n} | {r['diff']:+.4f} | {r['ci_low']:.4f} إلى {r['ci_high']:.4f} | {r['margin']} | {r['status']} |"
        for n, r in guard.iterrows())
    md = f"""# تقرير التجربة: عرض تقييمات المستخدمين بجانب المنتجات

**القرار: {decision}.** {reason}

## النتيجة الرئيسية
| البند | القيمة |
|---|---|
| نسبة الشراء | A: {pct(prim['mean_A'])} ← B: {pct(prim['mean_B'])} |
| الأثر | {pts(prim['diff'])} (فترة الثقة 95%: {prim['ci_low'] * 100:.2f} إلى {prim['ci_high'] * 100:.2f})، p = {prim['p']:.2g} |
| الأثر مع التعديل بالانحدار | {pts(adj['diff'])} ({adj['ci_low'] * 100:.2f} إلى {adj['ci_high'] * 100:.2f}) |
| الزيادة النسبية | {prim['relative_lift']:.1%} |
| الإيراد لكل مستخدم | {rev['diff']:+.2f} ريال (Bootstrap: {rev['boot_ci_low']:.2f} إلى {rev['boot_ci_high']:.2f}) |
| الربح السنوي التقديري | {annual['diff']:.2f} مليون ريال ({annual['ci_low']:.2f} إلى {annual['ci_high']:.2f}) |
| فحص SRM | A={s['n_A']:,}، B={s['n_B']:,}، p={s['p']:.3f} ({'سليم' if s['ok'] else 'خلل'}) |

## مقاييس الحماية (اختبار عدم الإضرار)
| المقياس | الفرق B − A | فترة الثقة 95% | الحد المسموح | الحالة |
|---|---|---|---|---|
{guard_rows}

## أثر الحداثة
الأثر في الأسبوع الأول {pts(week['week1']['diff'])} وفي الثاني {pts(week['week2']['diff'])}، والفرق بينهما غير دال
(p = {week['interaction_p']:.2f})، فلا دليل قوي على أن الأثر يختفي مع الوقت.

## القيود
- البيانات من محاكاة؛ الأثر الحقيقي المفترض 0.70 نقطة.
- مدة أسبوعين لا تمثل المواسم (رمضان، الجمعة البيضاء).
- لم نقس أثر التقييمات السلبية على منتجات بعينها، ولا أثر الميزة على البائعين.
"""
    (RESULTS_DIR / "report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تحليل تجربة عرض التقييمات بجانب المنتجات")
    parser.add_argument("--regenerate", action="store_true", help="إعادة إنشاء البيانات بالمحاكاة")
    main(parser.parse_args().regenerate)
