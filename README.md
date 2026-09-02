# Voxa — transcritor de áudio de sistema (Windows)

MVP desktop que captura o que o **Windows está reproduzindo** (WASAPI loopback), transcreve **localmente** com `faster-whisper` e mostra o texto em tempo real.

Não usa microfone. Não envia áudio para Google, Azure ou qualquer API online.

```text
audio_transcriber/
├── main.py              (Interface tkinter e fluxo principal)
├── audio_capture.py     (WASAPI loopback via sounddevice)
├── transcriber.py       (faster-whisper em thread)
├── utils.py             (tempo, resample, salvar)
└── requirements.txt
```

## Requisitos

- Windows 10/11
- Python 3.12
- Saída de áudio ativa (fones, caixas, HDMI…)
- Na **primeira** execução: internet só para baixar o modelo Whisper. Depois o app é 100% offline.

## Instalação

No PowerShell, dentro desta pasta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Se `py -3.12` não existir, use `python` desde que `python --version` seja 3.12.

## Uso

1. Abra o Voxa.
2. Idioma padrão: **pt**. Troque se precisar (en, es, fr, de, it ou auto).
3. Modelo: `tiny` (mais rápido), `base` (padrão), `small` (mais preciso, mais lento).
4. Clique em **Iniciar**. Na primeira vez o modelo é carregado.
5. Reproduza qualquer áudio no Windows (YouTube, reunião, filme).
6. O texto aparece com horário. **Salvar** gera um `.txt` UTF-8.

Se o áudio do sistema parar ou o dispositivo cair, o app **não fecha**. A barra de status mostra o erro e você clica em Iniciar de novo.

## Como funciona

| Peça | Papel |
| --- | --- |
| `audio_capture.py` | Abre o dispositivo de **saída** WASAPI em modo loopback, segmenta por silêncio (~0,8 s) ou no máximo 5 s |
| `transcriber.py` | `WhisperModel` em CPU `int8` (ou CUDA `float16` se houver GPU) |
| `main.py` | Tkinter na thread da UI; captura e whisper em threads; filas para não travar a janela |
| `utils.py` | Mono + resample 16 kHz, timestamps, gravação do `.txt` |

O callback do PortAudio só enfileira blocos. Whisper nunca roda na thread da interface.

## Observações

- Loopback WASAPI **não existe no Linux/macOS**. Este repositório é para Windows.
- Apps em modo exclusivo (alguns jogos e DAWs) podem bloquear o loopback. Feche-os ou desative o modo exclusivo no painel de som.
- CPU fraca: use o modelo `tiny`.
- Visual C++ Redistributable (x64) pode ser necessário para o `ctranslate2`.

## Licença

Uso livre neste repositório. Modelos Whisper seguem a licença original da OpenAI / Systran.
