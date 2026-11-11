"""أدوات عامة: طباعة منسقة، وتجهيز النص العربي للرسوم."""
import matplotlib
matplotlib.use("Agg")  # يرسم بدون شاشة، فيعمل في أي بيئة
from matplotlib import ft2font

_NATIVE = hasattr(ft2font, "__libraqm_version__")
if not _NATIVE:
    import arabic_reshaper
    from bidi.algorithm import get_display


def ar(text):
    """يعيد النص العربي بالشكل والاتجاه الصحيحين داخل رسوم matplotlib."""
    text = str(text)
    return text if _NATIVE else get_display(arabic_reshaper.reshape(text))


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def pts(x, d=2):
    """فرق نسبتين بالنقاط المئوية."""
    return f"{x * 100:+.{d}f} نقطة"
