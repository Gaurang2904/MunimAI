#!/bin/sh
# Convert the merged fine-tuned model to GGUF and register it with Ollama as "munim-nlu" (f16).
# Run from the MunimAI folder after finetune/train_nlu.py. Uses VehicleGPT's llama.cpp checkout and Python env.
set -e
PY=../VehicleGPT/.venv-ft/Scripts/python.exe
mkdir -p models
$PY ../VehicleGPT/tools/llama.cpp/convert_hf_to_gguf.py finetune/out/merged \
    --outfile models/munim-nlu-f16.gguf --outtype f16
printf 'FROM ./munim-nlu-f16.gguf\nPARAMETER temperature 0\n' > models/Modelfile
cd models && ollama create munim-nlu -f Modelfile
