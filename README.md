# Voxa — transcritor de áudio de sistema (Windows)

MVP desktop que captura o que o **Windows está reproduzindo** (WASAPI loopback via **PyAudioWPatch**), reamostra para **16 kHz mono** e transcreve **localmente** com `faster-whisper`.

Não usa microfone. Nenhuma API de transcrição online. O único acesso à rede é o **download único** dos pesos do modelo na primeira execução; daí em diante `local_files_only=True`.

```text
audio_transcriber/
├── main.py              (tkinter + threads)
├── audio_capture.py     (loopback + downmix + resample 16 kHz)
├── transcriber.py       (faster-whisper + dedup por timestamp)
├── utils.py             (diagnóstico de dispositivos loopback)
└── requirements.txt
```

## Requisitos

- Windows 10/11
- Python 3.12
- Saída de áudio ativa (fones, caixas, HDMI…)
- Visual C++ Redistributable x64 (o `ctranslate2` precisa)

## Instalação

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

`PyAudioWPatch` vem com **wheels pré-compilados** — não precisa de compilador C++ só para o loopback.

Diagnóstico bruto (opcional):

```powershell
python -m pyaudiowpatch
```

## Uso

1. **Diagnosticar** — lista só dispositivos **loopback**, nunca o microfone. A taxa nativa (quase sempre 48000 Hz) aparece no seletor.
2. Idioma padrão **pt**. Modelo `base` (ou `tiny` / `small`).
3. **Iniciar**. Primeira vez baixa o modelo; as seguintes são 100% locais.
4. Reproduza YouTube, reunião, filme. O app captura o playback, não o mic.
5. Latência esperada: **chunk 6 s + inferência** (tipicamente 8–14 s no total em CPU).
6. **Parar** grava um `.txt` UTF-8. **Salvar** escolhe o caminho.

Se o áudio cair ou o loopback sumir, a janela **não fecha** — a barra de status avisa.

## Pipeline

1. Diagnóstico `get_loopback_device_info_generator()` (PyAudioWPatch)
2. Captura na **taxa nativa** (ex.: 48000 Hz estéreo)
3. Downmix mono + `scipy.signal.resample` → **16000 Hz**
4. Chunk **6 s**, hop **4,5 s**, overlap **1,5 s**
5. `faster-whisper` `word_timestamps=True`, `compute_type="int8"`, `device="auto"`
6. Dedup: descarta palavras cujo `start` cai na janela já coberta pelo chunk anterior (não compara texto)
7. UI via `queue` + `root.after` — Whisper nunca roda na thread do tkinter

## Dependências

```
PyAudioWPatch==0.2.12.8
numpy==2.0.2
scipy==1.14.1
faster-whisper==1.1.1
```

## Licença

MIT. Modelos Whisper seguem a licença original da OpenAI / Systran.
