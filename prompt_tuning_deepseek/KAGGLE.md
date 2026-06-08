# Chạy trên Kaggle (đơn giản)

Giả định: bạn tải code này từ GitHub thẳng vào Kaggle Notebook.

## 1. Tạo notebook + bật GPU/Internet

- **Settings → Accelerator**: chọn `GPU T4 x2` (hoặc `P100`/`A100` nếu có quota)
- **Settings → Internet**: bật **On** (cần để `git clone`, `pip install`, tải model từ HuggingFace)

## 2. Clone repo

```python
!git clone https://github.com/trhuyyy13/prompt-tuning-api-migration /kaggle/working/repo
%cd /kaggle/working/repo
```

## 3. Cài thư viện

```python
!pip install -q -r prompt_tuning_deepseek/requirements.txt
```

## 4. Kiểm tra dataset

Nếu `data_raw/outdated_y+_FINAL.json` đã nằm trong repo đã clone thì khỏi làm gì thêm.
Nếu bạn để dataset dưới dạng **Kaggle Dataset** (Add Data ở sidebar), đường dẫn sẽ là
`/kaggle/input/<ten-dataset>/outdated_y+_FINAL.json` — chỉ cần trỏ `--train_file`/`--data_file`
tới đường dẫn đó thay vì `data_raw/...`.

## 5. Train

> ⚠️ GPU T4/P100 trên Kaggle **không hỗ trợ tốt `--bf16`** (cần kiến trúc Ampere trở lên như A100).
> Dùng **`--fp16`** thay vì `--bf16` trên T4/P100. Nếu bạn được cấp A100 thì dùng `--bf16` như bình thường.

```python
!python prompt_tuning_deepseek/train_prompt_tuning.py \
  --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
  --train_file data_raw/outdated_y+_FINAL.json \
  --valid_file data_raw/outdated_y+_FINAL.json \
  --output_dir /kaggle/working/outputs/prompt_tuning_deepseek_global \
  --num_virtual_tokens 20 \
  --prompt_init random \
  --max_input_length 512 \
  --max_target_length 128 \
  --max_seq_length 640 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-3 \
  --num_train_epochs 10 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --seed 42 \
  --fp16
```

Lưu **`--output_dir`** vào `/kaggle/working/...` (không phải `/kaggle/working/repo/...` nội bộ
là cũng được, miễn nằm trong `/kaggle/working`) để khi **Save Version** dữ liệu được giữ lại.
`soft_prompt.pt` rất nhỏ (vài trăm KB) nên không lo hết dung lượng output.

### Nếu bị ngắt phiên giữa chừng (session timeout ~9-12h)

Chạy lại đúng lệnh trên, thêm `--resume_from_checkpoint` (script tự đọc lại
`soft_prompt.pt` + `training_state.pt` trong `--output_dir`):

```python
!python prompt_tuning_deepseek/train_prompt_tuning.py \
  ... (các flag như trên) ... \
  --resume_from_checkpoint
```

## 6. Evaluate generation

```python
!python prompt_tuning_deepseek/evaluate.py \
  --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
  --data_file data_raw/outdated_y+_FINAL.json \
  --checkpoint_dir /kaggle/working/outputs/prompt_tuning_deepseek_global \
  --output_file /kaggle/working/outputs/prompt_tuning_deepseek_global/predictions.json \
  --max_input_length 512 \
  --max_new_tokens 128 \
  --limit 200
```

Bỏ `--limit` nếu muốn chạy full ~9k sample (sẽ lâu vì generate từng sample một —
xem README chính để biết lý do không batch generation với soft prompt).

## 7. Test forget quality / API migration quality

```python
!python prompt_tuning_deepseek/test_forget_quality.py \
  --predictions_file /kaggle/working/outputs/prompt_tuning_deepseek_global/predictions.json \
  --output_metrics /kaggle/working/outputs/prompt_tuning_deepseek_global/forget_quality_metrics.json \
  --output_details /kaggle/working/outputs/prompt_tuning_deepseek_global/forget_quality_details.json
```

Hoặc generate thẳng từ checkpoint (không cần chạy bước 6 trước, giống style
`Thamkhao/forget_quality.py` — tiện để bạn test lại trên chính tập đã train):

```python
!python prompt_tuning_deepseek/test_forget_quality.py \
  --checkpoint_dir /kaggle/working/outputs/prompt_tuning_deepseek_global \
  --data_file data_raw/outdated_y+_FINAL.json \
  --limit 500 \
  --output_metrics /kaggle/working/outputs/prompt_tuning_deepseek_global/forget_quality_metrics.json \
  --output_details /kaggle/working/outputs/prompt_tuning_deepseek_global/forget_quality_details.json
```

## 8. Push checkpoint lên Hugging Face Hub

Trước tiên tạo token tại https://huggingface.co/settings/tokens (chọn quyền **Write**).
Có 2 cách nhập token vào notebook — **chọn 1**:

### Cách 1 — Kaggle Secrets (khuyên dùng: token không hiện ra trong notebook/output)

- Mở **Add-ons → Secrets**, thêm secret tên `HF_TOKEN`, giá trị = token vừa tạo.
- Trong cell đầu tiên:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### Cách 2 — Nhập tay mỗi lần chạy (token chỉ tồn tại trong phiên hiện tại)

```python
import os, getpass
os.environ["HF_TOKEN"] = getpass.getpass("Nhập Hugging Face token (write): ")
```

### Push

`push_to_hub.py` tự đọc token từ biến môi trường `HF_TOKEN` (hoặc truyền `--token`
nếu muốn). Repo sẽ được tạo tự động nếu chưa tồn tại (mặc định public — thêm
`--private` nếu muốn riêng tư):

```python
!python prompt_tuning_deepseek/push_to_hub.py \
  --checkpoint_dir /kaggle/working/outputs/prompt_tuning_deepseek_global \
  --repo_id <ten-cua-ban>/depapi-soft-prompt-deepseek \
  --commit_message "soft prompt sau 10 epoch"
```

Toàn bộ thư mục `--checkpoint_dir` (`soft_prompt.pt`, tokenizer, `training_state.pt`,
`prompt_config.json`, `training_args.json`, ...) được upload nguyên trạng — đủ để
sau này tải lại bằng `huggingface_hub.snapshot_download(repo_id=...)` rồi trỏ
`--checkpoint_dir` của `evaluate.py` / `test_forget_quality.py` vào thư mục đã tải.

## Lưu ý nhanh cho Kaggle

- **fp16/bf16**: T4/P100 → `--fp16`; A100/A10/RTX30xx+ → `--bf16`.
- **OOM**: giảm `--per_device_train_batch_size` xuống 1 và tăng
  `--gradient_accumulation_steps` (giữ effective batch size như cũ).
- **Output**: luôn ghi `--output_dir` / `--output_file` vào `/kaggle/working/...`
  để giữ lại sau khi notebook kết thúc / Save Version.
- **Thời gian**: ưu tiên test nhanh bằng `--limit` ở bước evaluate/test trước,
  rồi mới chạy full nếu kết quả ổn.
