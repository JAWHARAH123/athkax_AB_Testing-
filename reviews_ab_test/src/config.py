"""كل الإعدادات في مكان واحد: افتراضات التصميم، و"الحقيقة" المخفية في المحاكاة، والمسارات."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"
CHARTS_DIR = ROOT / "outputs" / "charts"
RESULTS_DIR = ROOT / "outputs" / "results"

# ---------------- التصميم: ما نعرفه أو نقدره قبل التجربة ----------------
DESIGN = {
    "baseline_rate": 0.06,      # نسبة الشراء الحالية (6% من المستخدمين يشترون)
    "mde_abs": 0.006,           # أصغر أثر يهمنا تجاريًا: 0.6 نقطة مئوية (+10% نسبيًا)
    "alpha": 0.05,              # مستوى الدلالة
    "power": 0.80,              # القوة الإحصائية
    "daily_users": 5_000,       # مستخدمون جدد يدخلون التجربة يوميًا
    "treatment_share": 0.50,    # 50% للنسخة B
    "salt": "reviews-exp-2026", # ملح دالة التجزئة المستخدمة في التوزيع
    "annual_users": 5_000 * 365,
}

# حدود مقاييس الحماية (عدم الإضرار): نرفض الإطلاق إذا تجاوزت فترة الثقة هذه الحدود
GUARDRAILS = {
    "profit":            {"label": "الربح لكل مستخدم (ريال)",   "bad": "down", "margin": 0.0},
    "cancelled":         {"label": "نسبة إلغاء الطلبات",        "bad": "up",   "margin": 0.010},
    "contacted_support": {"label": "نسبة التواصل مع الدعم",      "bad": "up",   "margin": 0.005},
    "page_load_ms":      {"label": "زمن تحميل الصفحة (ملي ثانية)", "bad": "up",   "margin": 100.0},
}

# ---------------- المحاكاة: "الحقيقة" التي لا يعرفها المحلل في الواقع ----------------
SIMULATION = {
    "seed": 42,
    "true_lift_abs": 0.007,         # الأثر الحقيقي للتقييمات على احتمال الشراء
    "novelty_extra": 0.25,          # زيادة مؤقتة في الأيام الأولى (فضول)
    "mobile_share": 0.70,
    "returning_share": 0.40,
    "order_value_median": 170,      # الوسيط بالريال
    "order_value_effect": 0.015,    # المشتري الواثق يختار منتجًا أغلى قليلًا (+1.5%)
    "margin": 0.25,                 # هامش الربح
    "cancel_rate": 0.06, "cancel_effect": -0.004,
    "support_rate": 0.03, "support_effect": 0.0,
    "load_ms_mean": 1200, "load_ms_mobile_extra": 300, "load_ms_effect": 40,
    "duplicate_share": 0.003,       # سجلات مكررة بسبب إعادة إرسال الأحداث
}
