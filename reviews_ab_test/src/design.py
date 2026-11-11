"""الخطوة 1: حجم العينة والمدة وأصغر أثر قابل للاكتشاف."""
import math

from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def sample_size_formula(p1, p2, alpha, power):
    """الصيغة اليدوية لمقارنة نسبتين (اختبار ذو طرفين)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


def sample_size_statsmodels(p1, p2, alpha, power):
    """تحقق ثانٍ بمكتبة statsmodels باستخدام حجم الأثر h لكوهين."""
    h = abs(proportion_effectsize(p2, p1))
    return math.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0))


def mde_for_n(p, n_per_group, alpha, power):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * math.sqrt(2 * p * (1 - p) / n_per_group)


def plan(design):
    p1 = design["baseline_rate"]
    p2 = p1 + design["mde_abs"]
    n_formula = sample_size_formula(p1, p2, design["alpha"], design["power"])
    n_sm = sample_size_statsmodels(p1, p2, design["alpha"], design["power"])
    n = max(n_formula, n_sm)
    raw_days = math.ceil(2 * n / design["daily_users"])
    days = math.ceil(raw_days / 7) * 7          # أسابيع كاملة لتغطية كل أيام الأسبوع
    n_expected = days * design["daily_users"] // 2
    return {
        "n_per_group_formula": n_formula,
        "n_per_group_statsmodels": n_sm,
        "n_per_group": n,
        "raw_days": raw_days,
        "days": days,
        "n_expected_per_group": n_expected,
        "mde_planned": mde_for_n(p1, n_expected, design["alpha"], design["power"]),
        "mde_one_week": mde_for_n(p1, 7 * design["daily_users"] // 2, design["alpha"], design["power"]),
    }
