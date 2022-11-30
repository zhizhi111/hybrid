import torch
from model.vqa import UniterForVisualQuestionAnswering
from tqdm import tqdm
import json
cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
with open("vocab.txt", "r") as f:
    vocab = f.read().splitlines()
root = "vqa_output_4tokens-mask"
checkpoint = torch.load(f"{root}/ckpt/model_step_20000.pt")
model = UniterForVisualQuestionAnswering.from_pretrained(f"{root}/log/model.json", checkpoint,img_dim=2048,num_answer=3129)
model.cuda()
word_dict = model.uniter.embeddings.word_embeddings

for name, param in model.named_parameters():
    if name == "uniter.soft_prompt":
        soft_prompt = param

templates = []
for i in tqdm(range(25)):
    template = []
    for j in tqdm(range(4)):
        matched_id = 0
        similarity = 1e-10
        for id in range(28996):
            s = cos(soft_prompt[i][j], word_dict(torch.tensor([id]).cuda()))
            if s > similarity:
                matched_id = id
                similarity = s
        template.append(matched_id)
    templates.append(template)

for i in range(25):
    n1, n2, n3, n4 = templates[i][0],templates[i][1],templates[i][2],templates[i][3]
    print(f"{i+1} {vocab[n1]}: {vocab[n2]}: {vocab[n3]}: {vocab[n4]}")