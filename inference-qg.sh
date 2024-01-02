base_model_repo="mistralai/Mistral-7B-Instruct-v0.1"
lora_repo="reinforz/mistral7b-instruct-lora-nf4-qg"
lora_file="lit_model_lora_finetuned.pth"
data_file_url=https://cdn.discordapp.com/attachments/1003310779157725194/1190684270084247622/test.json

cd workspace

git clone https://github.com/Reinforz/lit-gpt .
git pull
pip install -r requirements-all.txt
pip uninstall -y torch torchvision torchaudio torchtext
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

curl -s $data_file_url -o data.json
python scripts/download.py \
  --repo_id $base_model_repo

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$base_model_repo

python scripts/infer.py \
  --lora_repo $lora_repo \
  --checkpoint_dir checkpoints/$base_model_repo \
  --data_dir data.json \
  --lora_file $lora_file