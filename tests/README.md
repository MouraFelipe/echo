# Testes — Voxa

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pip install numpy scipy
py -3.12 -m pytest
```

A suíte **não** abre WASAPI nem carrega Whisper. Hardware, `.exe` e o modelo real entram só nos testes manuais / de integração marcados.

| Arquivo | O que cobre |
|---|---|
| `test_utils.py` | tempo, resample 16 kHz, RMS, save UTF-8 BOM, loopback fora do Windows |
| `test_devices.py` | diagnóstico WASAPI com fake PyAudioWPatch |
| `test_audio_capture.py` | open/read/stop, chunk 6 s + hop 4,5 s + overlap 1,5 s |
| `test_transcriber.py` | dedup por timestamp, join, peak-norm, idiomas/modelos |
| `test_assistant.py` | resumo/tarefas/pergunta + Ollama opcional |
| `test_concurrency.py` | fila maxsize=3, loop de transcrição, double-start |
| `test_persistence.py` | Parar/salvar/copiar |
| `test_runtime_hook.py` | cache HF no `.exe` |
