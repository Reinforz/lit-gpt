hf_repo_author="tiiuae"
hf_repo_name="falcon-7b"
dir_name="thesis"
# https://cdn.discordapp.com/attachments/1003310779157725194/1190165631086120960/data.json for qg data
# https://cdn.discordapp.com/attachments/833018351009529856/1189824144754352188/data.json for eval data
data_file_url="https://cdn.discordapp.com/attachments/1003310779157725194/1190563478814072892/train.json"

cd workspace
mkdir $dir_name
cd $dir_name
git clone https://github.com/Reinforz/lit-gpt .
pip install -r requirements-all.txt
python scripts/download.py \
  --repo_id $hf_repo_author/$hf_repo_name
python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/$hf_repo_author/$hf_repo_name
curl -s $data_file_url -o /$dir_name/data.json
python scripts/prepare_alpaca.py \
  --data_file_name data.json --destination_path /$dir_name --checkpoint_dir checkpoints/$hf_repo_author/$hf_repo_name
python finetune/lora.py \
  --checkpoint_dir checkpoints/$hf_repo_author/$hf_repo_name/ --precision bf16-true --quantize "bnb.nf4" --data_dir /$dir_name --out_dir out/$hf_repo_author --repo_id reinforz/falcon7b-instruct-lora-nf4-subj-eval