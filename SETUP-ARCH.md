# Настройка окружения (Arch Linux)

Две машины, один репозиторий:

| Машина | Железо | Роль | Torch |
|---|---|---|---|
| Ноутбук | AMD iGPU | разработка, тесты, librosa, экспорт | CPU |
| ПК | RTX 3050 8 GB | обучение, Demucs, инференс | CUDA |

`nvidia-smi` на ноутбуке падает — **это нормально**, там нет NVIDIA.
Проверять CUDA нужно только на ПК.

---

## 0. Если всё это впервые

Порядок действий и что означает каждая команда. Понимать всё до последней
буквы не нужно — достаточно знать, что происходит.

| Команда | Что делает |
|---|---|
| `sudo pacman -S --needed ffmpeg uv` | ставит программы в систему (Arch) |
| `cd ~/Projects/SoundToTab` | переходит в папку проекта |
| `uv venv --python 3.12` | создаёт `.venv/` — отдельную папку с Python 3.12 и пакетами только для этого проекта |
| `source .venv/bin/activate` | «включает» окружение в текущем терминале (в начале строки появится `(.venv)`) |
| `uv pip install torch` | ставит пакет внутрь `.venv`, а не в систему |
| `python -c "..."` | выполняет одну строку Python |
| `git init` | превращает папку в репозиторий с историей версий |
| `git add .` + `git commit -m "..."` | сохраняет текущее состояние в историю |

Три вещи, которые сбивают с толку чаще всего:

1. **`(.venv)` в приглашении терминала.** Нет скобок — окружение не
   активировано, и `python` будет системный. Лечится повторным
   `source .venv/bin/activate` из папки проекта.
2. **Системный Python 3.14.** Он новее, чем поддерживает PyTorch, поэтому
   проект живёт на отдельном Python 3.12. `uv` скачивает его сам, `pacman`
   для этого не нужен. Системный Python не трогаем.
3. **`pip` в системе не установлен, и это правильно.** Ставить пакеты в
   системный Python на Arch нельзя — только в `.venv`.

Проверка, что окружение активно:

```bash
which python        # должно быть .../SoundToTab/.venv/bin/python
python --version    # Python 3.12.x
```

---

## 1. Базовые пакеты

### Ноутбук

```bash
sudo pacman -S --needed base-devel git ffmpeg libsndfile python uv
```

`uv` есть в extra. Если пакета нет:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### ПК

То же самое + драйвер:

```bash
sudo pacman -S --needed base-devel git ffmpeg libsndfile python uv
```

Если `nvidia-smi` не работает:

```bash
# обычное ядро
sudo pacman -S --needed nvidia nvidia-utils
# или, если стоит linux-lts / кастомное ядро
sudo pacman -S --needed nvidia-dkms nvidia-utils
```

Затем перезагрузка и проверка:

```bash
nvidia-smi
```

Для гибридных ноутбуков с NVIDIA дополнительно `nvidia-prime` и запуск через
`prime-run`. Для десктопного ПК не нужно.

---

## 2. Проект и виртуальное окружение

На **обеих** машинах:

```bash
cd ~/Projects/SoundToTab
uv venv --python 3.12
source .venv/bin/activate
```

`.venv/` в `.gitignore`, в репозиторий не попадает.

---

## 3. PyTorch

Индекс CUDA берём актуальный с <https://pytorch.org/get-started/locally/> —
ниже пример, версия может отличаться.

### Ноутбук (CPU)

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### CUDA на ПК

Отдельная задача на один вечер, делать до недели 6 — перед первым обучением
модели. На ноутбуке этот раздел не нужен.

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Проверка

```bash
# ноутбук: ожидаем False и пустое имя
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ПК: ожидаем True и NVIDIA GeForce RTX 3050 ...
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Если на ПК `False` — почти всегда поставился CPU-колесо. Проверить:

```bash
python -c "import torch; print(torch.version.cuda)"
```

`None` → переустановить с CUDA-индексом.

---

## 4. Зависимости проекта

На неделе 0 нужны только `torch`/`torchaudio` (раздел 3), `librosa` и
`soundfile`. Остальное — по мере надобности, начиная с недели 1.

Лёгкие (ставятся на обеих машинах):

```bash
uv pip install numpy scipy librosa soundfile mir_eval pretty_midi pyguitarpro typer rich
```

Тяжёлые (можно только на ПК, но на ноуте тоже полезны для тестов на коротких
файлах):

```bash
uv pip install demucs basic-pitch
```

Важные оговорки:

- **`demucs` тянет свои версии torch/torchaudio** и может заменить CUDA-сборку
  на CPU. После установки снова проверь `torch.version.cuda`; если `None` —
  переустанови torch с CUDA-индексом.
- **`basic-pitch` тянет TensorFlow** (~2 ГБ). Если не хочется — есть ONNX-вариант
  инференса, но начинать проще с обычного пакета.
- `pretty_midi` для записи MIDI ничего дополнительного не требует;
  `fluidsynth` нужен только если захочешь синтезировать звук.

---

## 5. Одинаковое окружение на двух машинах

Пока `pyproject.toml` и `uv.lock` нет (появятся на неделе 8), источник правды —
этот файл. На каждой машине выполняются одни и те же команды из разделов 3–4,
поэтому окружения получаются одинаковыми, кроме сборки torch.

Когда появится `uv.lock`: `uv sync` поставит всё, кроме правильной сборки torch,
и её нужно будет доустановить вручную нужным индексом. Лок-файл при этом
коммитим, `.venv/` — нет.

```bash
git add pyproject.toml uv.lock
git commit -m "chore: lock dependencies"
```

---

## 6. Проверка «всё живое»

```bash
ffmpeg -version | head -n 1
python -c "import librosa, soundfile, pretty_midi, mir_eval; print('deps ok')"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Быстрый тест на реальном файле (ноутбук, любой mp3):

```bash
python - <<'PY'
import librosa
y, sr = librosa.load("test.mp3", sr=22050, mono=True)
print(y.shape, sr)
m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print(m.shape)
PY
```

Если это отработало — окружение готово.

---

## 7. Типовые проблемы

| Симптом | Причина | Что делать |
|---|---|---|
| `nvidia-smi` падает на ноуте | нет NVIDIA | нормально, игнорировать |
| `nvidia-smi` падает на ПК | драйвер/модуль ядра | `nvidia-dkms` + reboot, проверить `dmesg` |
| `torch.cuda.is_available()` → False на ПК | CPU-колесо | переустановить с CUDA-индексом |
| `librosa` не читает mp3 | нет ffmpeg | `sudo pacman -S ffmpeg` |
| `soundfile` не открывает файл | нет libsndfile | `sudo pacman -S libsndfile` |
| demucs ругается на torchaudio | несовпадение версий | переустановить пару torch+torchaudio одним индексом |
| CUDA out of memory на 3050 | большие сегменты/батчи | `--segment 20`, батч 8–16, `torch.autocast` |
| Медленно на ноуте | CPU-инференс Demucs | тестировать на 30-секундных отрывках |

---

## 8. Полезные привычки

- На ноуте запускать короткие отрывки (10–30 с), полные треки — на ПК.
- Все пути — через `pathlib.Path`, без абсолютных путей к домашней папке:
  репозиторий должен работать на обеих машинах.
- `device = "cuda" if torch.cuda.is_available() else "cpu"` — в одном месте,
  а не по всему коду.
- Перед `git push` проверять `git status`: `.venv/`, `data/`, `*.wav`, `*.gp5`
  не должны попадать в индекс.
