# Configuração — Echo

## Requisitos

- Windows 10 ou 11 (WASAPI loopback não existe no Linux/macOS)
- Python 3.12 se for rodar do código; o `.exe` já traz o runtime
- Visual C++ Redistributable x64 se o Windows pedir DLL

Não precisa de FFmpeg, chave de API nem microfone.

## Modelo Whisper

Na primeira execução o Echo baixa o modelo (`tiny` / `base` / `small`) para `%LOCALAPPDATA%\Echo\hf`. Depois usa `local_files_only=True` e funciona offline.

## Assistente

Sem variável de ambiente. Se `http://127.0.0.1:11434` responder (Ollama), o Echo usa o primeiro modelo instalado. Senão, resumidor extrativo local.

## Testes

```powershell
pip install -r requirements-dev.txt numpy scipy
python -m pytest
```

A suíte não abre WASAPI nem carrega Whisper.
