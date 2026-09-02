# Voxa — transcritor de áudio de sistema (Windows)

MVP desktop que captura o que o **Windows está reproduzindo** (WASAPI loopback via **PyAudioWPatch**), reamostra para **16 kHz mono** e transcreve **localmente** com `faster-whisper`.

Não usa microfone. Nenhuma API de transcrição online. O único acesso à rede é o **download único** dos pesos do modelo na primeira execução; daí em diante `local_files_only=True`.

## Baixar o .exe (recomendado)

Não precisa instalar Python.

1. Baixe **[Voxa.exe](https://github.com/MouraFelipe/audio-transcriber/releases/latest/download/Voxa.exe)**
2. Dê dois cliques. O SmartScreen pode avisar (app sem certificado) — *Mais informações → Executar mesmo assim*.
3. Na primeira execução o modelo Whisper é baixado (~150 MB) para `%LOCALAPPDATA%\Voxa\hf`. Depois funciona offline.
4. Se faltar DLL, instale o [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe).

O GitHub Actions gera esse `.exe` a cada push em `main` (`voxa.spec` + PyInstaller onefile).

## Rodar do código (opcional)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Gerar o `.exe` localmente no Windows:

```powershell
pip install pyinstaller==6.11.1
pyinstaller --noconfirm --clean voxa.spec
# sai em dist\Voxa.exe
```

## Uso

1. **Diagnosticar** — lista só dispositivos **loopback**, nunca o microfone.
2. Idioma padrão **pt**. Modelo `base` (ou `tiny` / `small`).
3. **Iniciar**. Reproduza YouTube, reunião, filme.
4. Latência esperada: **chunk 6 s + inferência** (tipicamente 8–14 s no `base` em CPU).
5. **Parar** grava um `.txt` UTF-8 ao lado do executável (`transcripts\`).

Se o áudio cair ou o loopback sumir, a janela **não fecha** — a barra de status avisa.

## Pipeline

1. Diagnóstico `get_loopback_device_info_generator()` (PyAudioWPatch)
2. Captura na **taxa nativa** (ex.: 48000 Hz estéreo)
3. Downmix mono + `scipy.signal.resample` → **16000 Hz**
4. Chunk **6 s**, hop **4,5 s**, overlap **1,5 s**
5. `faster-whisper` `word_timestamps=True`, `compute_type="int8"`
6. Dedup por timestamp de palavra (não por texto)

## Licença

MIT. Modelos Whisper seguem a licença original da OpenAI / Systran.
