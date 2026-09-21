# Guia de uso — Echo

O Echo captura o que o **Windows está reproduzindo** (YouTube, reunião, filme) e transcreve localmente. Não usa microfone.

## Pelo Echo.exe (recomendado)

1. Baixe o [Echo.exe](https://github.com/MouraFelipe/echo/releases/latest/download/Echo.exe) e dê dois cliques.
2. **Diagnosticar** — lista só dispositivos loopback. Escolha o mesmo da saída de som (fone/caixa), não o mic.
3. Idioma `pt`, modelo `base` (ou `tiny` / `small`).
4. **Iniciar** e reproduza o áudio. O texto aparece com ~8–14 s de atraso.
5. **Parar** grava um `.txt` em `transcripts\`. **Salvar** também oferece `.srt`.
6. **Resumir / Tarefas / Perguntar** — assistente local. Se o Ollama estiver aberto neste PC, ele entra no lugar do resumidor.

Se o YouTube estiver no HDMI e o Echo no Realtek, a transcrição vem vazia: os dois precisam ser o mesmo dispositivo.

## Pelo código

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
