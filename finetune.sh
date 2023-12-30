hf_repo="mistralai/Mistral-7B-Instruct-v0.1"
# https://cdn.discordapp.com/attachments/1003310779157725194/1190165631086120960/data.json for qg data
# https://cdn.discordapp.com/attachments/833018351009529856/1189824144754352188/data.json for eval data
data_file_url="https://cdn.discordapp.com/attachments/1003310779157725194/1190563478814072892/train.json"

cd workspace
git clone https://github.com/Reinforz/lit-gpt .
pip install -r requirements-all.txt
python scripts/download.py \
  --repo_id $hf_repo
python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/$hf_repo
curl -s $data_file_url -o data.json
python scripts/prepare_alpaca.py \
  --data_file_name data.json --destination_path . --checkpoint_dir checkpoints/$hf_repo
# python finetune/lora.py \
#   --checkpoint_dir checkpoints/$hf_repo/ --precision bf16-true --quantize "bnb.nf4" --data_dir . --out_dir out --repo_id reinforz/falcon7b-instruct-lora-nf4-subj-eval
python finetune/adapter_v2.py \
  --checkpoint_dir checkpoints/$hf_repo/ --precision bf16-true --data_dir . --out_dir out --repo_id reinforz/mistral7b-instruct-adapterv2-nf4-subj-eval