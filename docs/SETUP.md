# راهنمای نصب و اجرا روی Windows 10 (LUNA AI)

## ۱. پیش‌نیازهای سیستم هدف
- سیستم‌عامل: **Windows 10 (64-bit)**
- رم: حداقل **4GB** (پیشنهادی: **8GB**)
- پردازنده: Intel Core i5 نسل سوم یا جدیدتر (بدون نیاز به GPU مجزا)
- فضای خالی دیسک: حدود **2GB** (برای Python، مرورگر سبک Chromium و مدل `qwen3:0.6b` با حجم ~522MB)
- نصب بودن **Python 3.10+** (گزینه *Add Python to PATH* در زمان نصب تیک خورده باشد)
- نصب بودن **Ollama for Windows** از آدرس رسمی: `https://ollama.com/download/windows`

---

## ۲. نصب خودکار در ویندوز ۱۰ (یک کلیک)
۱. پوشه `LUNA` را در مسیر دلخواه قرار دهید.
۲. فایل **`setup_windows.bat`** را اجرا کنید. این اسکریپت به‌طور خودکار:
   - محیط مجازی `.venv` را می‌سازد،
   - کتابخانه‌های سبک `requirements.txt` را نصب می‌کند،
   - مرورگر Chromium مورد نیاز Playwright را نصب می‌کند،
   - مدل محلی `qwen3:0.6b` را در Ollama دریافت می‌کند.

---

## ۳. نصب دستی (گام‌به‌گام در PowerShell یا CMD)
```bat
cd LUNA
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
ollama pull qwen3:0.6b
```

---

## ۴. اجرای برنامه
کافی است روی **`run.bat`** دابل‌کلیک کنید (یا دستور زیر را بزنید):
```bat
python main.py --host 127.0.0.1 --port 8000
```
سپس مرورگر روی آدرس زیر باز می‌شود:
`http://127.0.0.1:8000`

### حالت‌های دیگر خط فرمان:
- بررسی واقعی وضعیت سیستم و سلامت Ollama:
  ```bat
  python main.py --check-env
  ```
- گفتگوی متنی در ترمینال (بدون مرورگر):
  ```bat
  python main.py --cli
  ```

---

## ۵. اجرای تست‌های خودکار و ارزیابی
```bat
pytest -v
pytest -m "not browser" -v
python training/evaluate.py --mode local_strict
python training/evaluate.py --mode hybrid_teacher
```
