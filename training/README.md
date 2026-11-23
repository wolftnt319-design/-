# آزمایشگاه اختیاری تنظیم پارامتر و آموزش مدل (LUNA Training & Evaluation Lab)

این زیرسیستم به‌صورت **کاملاً جداگانه از مسیر اجرای اصلی برنامه** طراحی شده است تا حافظه و پردازنده در زمان کار عادی کاربر اشغال نشود.

## محدودیت‌های سخت‌افزاری هدف
- سیستم هدف: Windows 10 | RAM: 8GB | CPU: Intel Core i5 نسل سوم | GPU: Intel HD Graphics (بدون کارت گرافیک مجزا و بدون CUDA).
- آموزش مستقیم وزن‌های `Qwen3-0.6B` روی CPU نسل سوم بسیار کند و پرمصرف است؛ بنابراین در LUNA ابتدا **مسیر A** توصیه و فعال شده است:
  1. تنظیم دقیق پارامترهای استنتاج جداگانه برای فارسی و انگلیسی (`temperature`, `top_p`, `top_k`, `repeat_penalty`, `presence_penalty`, `num_ctx`, `seed`, `stop`) در `data/settings.json` و پنل گرافیکی.
  2. بازیابی هیبرید حافظه بلندمدت (`SQLite FTS5 + char_wb TF-IDF + RRF`).
  3. تقطیر دانش در لحظه (In-Context Knowledge Distillation) از طریق نمونه پاسخ‌های تأییدشده در جدول `teacher_exemplars`.

## مسیر B: آماده‌سازی دیتاست و آموزش LoRA اختیاری

### ۱. ساخت دیتاست زمانی (بدون نشت داده مکالمه)
```bash
python training/prepare_dataset.py --output-dir data/datasets
```
- تمام رمزها، کارت‌های بانکی و توکن‌ها با `SecretRedactor` حذف یا رد می‌شوند.
- تقسیم داده‌ها به‌صورت **زمانی (Chronological Split: 70% Train / 15% Val / 15% Test)** انجام می‌شود، نه `train_test_split` تصادفی.

### ۲. اجرای ارزیابی استاندارد فارسی و انگلیسی (JSON + CSV)
```bash
python training/evaluate.py --mode local_strict
python training/evaluate.py --mode hybrid_teacher
```
خروجی در `data/evaluations/` به‌صورت فایل‌های `.json` و `.csv` ذخیره می‌شود.

### ۳. اجرای آموزش LoRA (فقط با تأیید صریح کاربر و در زمان بیکاری Ollama)
```bash
python training/train_lora.py --dataset-dir data/datasets --confirm-user-approval
```
- نیازمند نصب اختیاری `torch`, `transformers`, `peft`, `datasets` است.
- در صورت فعال بودن مدل در حافظه Ollama، پیش از شروع هشدار داده و متوقف می‌شود.
- فقط وزن‌های سبک Adapter ذخیره شده و امکان بازگشت (`--rollback`) فراهم است.
