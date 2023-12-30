base_model_repo="mistralai/Mistral-7B-Instruct-v0.2"
lora_repo="reinforz/mistral7b-instruct-lora-nf4-subj-eval"

git clone https://github.com/Reinforz/lit-gpt .
pip install -r requirements-all.txt
cd thesis
python scripts/download.py \
  --repo_id $base_model_repo

huggingface-cli login --token $HUGGINGFACE_TOKEN

huggingface-cli download $lora_repo --output out

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$base_model_repo

python scripts/merge_lora.py \
  --lora_path out/lit_model_lora_finetuned.pth --checkpoint_dir checkpoints/$base_model_repo --out_dir out/lora/checkpoint

python generate/lora.py \
  --prompt "Instructions: Evaluate the answer of the following question. Give a score in terms of relevence, coherene and grammar and explanation of for the evaluation. Please structure your response as follows:\n1. Begin with the 'Answer Evaluation' section, offering an in-depth review and analysis of the student's answer with respect to the given evaluation criteria.\n2. Follow this with a whole number numerical score for relevance (out of 6), coherence (out of 2), and grammar & spelling (out of 2) for the student's answer. Each score should be listed on a new line, preceded by its respective category." \
  --input "Question: How does 3D visualization contribute to geographical techniques and analysis? Discuss its implications and challenges, as well as possible future directions.\nEvaluation Criteria: The answer should begin by explaining what 3D visualization is and its role in geographical techniques.\nThe response should then elaborate on the implications of 3D visualization in geographical analysis, utilizing real-world examples for illustration.\nThe student should delve into the challenges of 3D visualization in this context, potentially drawing from academic literature or case studies.\nThe discussion should also encompass potential future directions and advancements in the field of 3D visualization and geographical analysis.\nThe language used should be formal and academic, with clear and concise sentences that are free from grammatical and spelling errors.\nThe answer should be logically structured, with a clear progression of ideas and arguments, and it should demonstrate a strong understanding of the topic.\nAnswer: 3D visualization is a powerful tool in geographical techniques, facilitating the representation of spatial data in a manner that closely mirrors real-world conditions. It enhances data interpretation by providing a more realistic and in-depth view of geographical features, thereby enriching spatial analysis and decision-making processes.\n\nOne notable implication of 3D visualization in geographical analysis is its ability to facilitate the comprehension of complex spatial patterns and relationships. For instance, in urban planning, 3D visualization can be used to model and analyze the spatial distribution of various urban features, such as population density, infrastructure, and land use. This allows planners to make more informed decisions about urban development.\n\nHowever, 3D visualization also presents certain challenges. The process of creating accurate and useful 3D models requires high-quality data, which can be difficult and expensive to obtain. Additionally, 3D models can be complex and require specialized skills to interpret and use effectively.\n\nIn terms of future directions, advancements in technology, such as improvements in data collection methods and computational power, are likely to make 3D visualization more accessible and useful. Developments in virtual and augmented reality technologies could also provide new opportunities for interactive 3D visualization, further enhancing our ability to understand and interpret spatial data.\n\nIn conclusion, while 3D visualization presents certain challenges, its potential benefits to geographical analysis are significant, and ongoing advancements in technology are likely to further enhance its usefulness in the future." \
  --lora_path out/lit_model_lora_finetuned.pth \
  --checkpoint_dir checkpoints/$base_model_repo \
  --quantize bnb.nf4 \
  --max_new_tokens 250