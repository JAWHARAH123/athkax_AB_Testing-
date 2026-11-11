"""الخطوة 5: الرسوم، وتحفظ في outputs/charts."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import ar

TEAL, ORANGE, NAVY, GRAY, RED = "#0E7490", "#E58A1F", "#0F2A3F", "#A9B4BC", "#B42318"
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.color": "#E3E8EC",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})


def conversion_bars(prim, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    means = [prim["mean_A"], prim["mean_B"]]
    labels = [ar("A: بدون تقييمات"), ar("B: مع التقييمات")]
    ax.bar(labels, np.array(means) * 100, color=[GRAY, TEAL], width=0.55)
    for i, m in enumerate(means):
        ax.text(i, m * 100 + 0.1, f"{m * 100:.2f}%", ha="center", color=NAVY, fontweight="bold")
    ax.set_ylabel(ar("نسبة الشراء (%)"))
    ax.set_title(ar("نسبة الشراء في كل نسخة"))
    ax.set_ylim(0, max(means) * 100 * 1.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def effect_forest(prim, adj, mde, path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rows = [(ar("مع التعديل بالانحدار (CUPED)"), adj), (ar("بدون تعديل"), prim)]
    for i, (label, r) in enumerate(rows):
        ax.errorbar(r["diff"] * 100, i, xerr=[[(r["diff"] - r["ci_low"]) * 100], [(r["ci_high"] - r["diff"]) * 100]],
                    fmt="o", color=TEAL, capsize=6, markersize=8, linewidth=2)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.axvline(mde * 100, color=ORANGE, linestyle="--", label=ar("أصغر أثر يهمنا"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(ar("الأثر على نسبة الشراء (نقطة مئوية) مع فترة ثقة 95%"))
    ax.set_title(ar("الأثر المقدر للتقييمات"))
    ax.set_ylim(-0.7, len(rows) - 0.3); ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def daily_effect(daily, overall, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(daily["day"], daily["ci_low"] * 100, daily["ci_high"] * 100, color=TEAL, alpha=0.15,
                    label=ar("فترة الثقة 95% لكل يوم"))
    ax.plot(daily["day"], daily["diff"] * 100, marker="o", color=TEAL, label=ar("الأثر في كل يوم"))
    ax.axhline(overall * 100, color=ORANGE, linestyle="--", label=ar("الأثر الكلي"))
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xlabel(ar("يوم دخول المستخدم")); ax.set_ylabel(ar("الأثر (نقطة مئوية)"))
    ax.set_title(ar("هل الأثر ثابت مع الوقت؟ (فحص أثر الحداثة)"))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def guardrail_chart(guard, path):
    fig, axes = plt.subplots(1, len(guard), figsize=(12, 3.4))
    for ax, (name, r) in zip(axes, guard.iterrows()):
        color = TEAL if r["status"] == "سليم" else RED
        ax.errorbar([r["diff"]], [0], xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                    fmt="o", color=color, capsize=6, linewidth=2)
        ax.axvline(0, color=GRAY, linewidth=1)
        limit = r["margin"] if r["bad_direction"] == "up" else -r["margin"]
        ax.axvline(limit, color=RED, linestyle=":", label=ar("الحد المسموح"))
        ax.set_yticks([]); ax.set_title(ar(name), fontsize=10)
        ax.set_xlabel(ar("الفرق B − A"), fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(ar("مقاييس الحماية"), fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def aa_hist(pvals, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(pvals, bins=20, range=(0, 1), color=GRAY, edgecolor="white")
    ax.axvline(0.05, color=RED, linestyle="--", label=ar(f"p < 0.05: {np.mean(pvals < 0.05):.1%} من الاختبارات"))
    ax.set_xlabel(ar("قيمة p")); ax.set_ylabel(ar("عدد الاختبارات"))
    ax.set_title(ar(f"{len(pvals)} اختبار A/A: توزيع قيم p متساوٍ تقريبًا كما يجب"))
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
