"""الخطوة 2: إنشاء بيانات التجربة بالمحاكاة وحفظها في data/dataset.csv.

كل صف = مستخدم دخل التجربة. التوزيع على النسختين يتم بدالة تجزئة (hash) على رقم المستخدم،
كما يحدث في أنظمة التجارب الحقيقية، فيرى المستخدم النسخة نفسها في كل زيارة.
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def assign_variant(user_ids, salt, treatment_share):
    """توزيع ثابت وقابل للتكرار: نحول (الملح + رقم المستخدم) إلى رقم بين 0 و1."""
    buckets = np.array([int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                        for u in user_ids])
    return np.where(buckets < treatment_share, "B", "A")


def _calibrate_intercept(score, target):
    """نختار ثابت المعادلة اللوجستية حتى يساوي متوسط احتمال الشراء خط الأساس."""
    f = lambda a: (1 / (1 + np.exp(-(a + score)))).mean() - target
    return brentq(f, -10, 5)


def generate(design, sim, days):
    rng = np.random.default_rng(sim["seed"])
    n = days * design["daily_users"]

    user_id = np.array([f"U{i:06d}" for i in range(1, n + 1)])
    variant = assign_variant(user_id, design["salt"], design["treatment_share"])
    is_b = (variant == "B").astype(int)
    day = rng.integers(1, days + 1, n)
    device = np.where(rng.random(n) < sim["mobile_share"], "mobile", "desktop")
    returning = (rng.random(n) < sim["returning_share"]).astype(int)

    # نشاط المستخدم الكامن (لا نقيسه)، ويظهر جزئيًا في طلباته السابقة
    activity = rng.normal(0, 1, n)
    pre_orders_90d = np.where(returning == 1, rng.poisson(np.exp(0.2 + 0.7 * activity)), 0)

    score = 0.6 * activity + 0.5 * returning - 0.2 * (device == "mobile")
    a = _calibrate_intercept(score, design["baseline_rate"])
    p0 = 1 / (1 + np.exp(-(a + score)))

    lift = sim["true_lift_abs"] * (1 + sim["novelty_extra"] * np.exp(-(day - 1) / 3))
    ordered = rng.binomial(1, np.clip(p0 + lift * is_b, 0, 1))

    value = np.exp(rng.normal(np.log(sim["order_value_median"]), 0.4, n)) * (1 + sim["order_value_effect"] * is_b)
    order_value = np.where(ordered == 1, value.round(2), np.nan)
    revenue = np.where(ordered == 1, value, 0.0).round(2)
    profit = (revenue * sim["margin"]).round(2)
    cancel_p = np.clip(sim["cancel_rate"] + sim["cancel_effect"] * is_b, 0, 1)
    cancelled = np.where(ordered == 1, rng.binomial(1, cancel_p), np.nan)
    support_p = np.clip(sim["support_rate"] + sim["support_effect"] * is_b, 0, 1)
    contacted_support = rng.binomial(1, support_p)
    load_mean = sim["load_ms_mean"] + sim["load_ms_mobile_extra"] * (device == "mobile") + sim["load_ms_effect"] * is_b
    page_load_ms = rng.gamma(shape=16, scale=load_mean / 16).round(0)

    df = pd.DataFrame({
        "user_id": user_id, "variant": variant, "day": day, "device": device, "returning": returning,
        "pre_orders_90d": pre_orders_90d, "ordered": ordered, "order_value": order_value,
        "revenue": revenue, "profit": profit, "cancelled": cancelled,
        "contacted_support": contacted_support, "page_load_ms": page_load_ms,
    })

    # سجلات مكررة كما يحدث في الواقع عند إعادة إرسال الأحداث، ونزيلها في خطوة التنظيف
    dup = df.sample(frac=sim["duplicate_share"], random_state=sim["seed"])
    df = pd.concat([df, dup]).sample(frac=1, random_state=sim["seed"]).reset_index(drop=True)
    return df


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
