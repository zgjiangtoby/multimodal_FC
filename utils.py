from torch.utils.data import Dataset
import clip, torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class Mocheg_Dataset(Dataset):
    def __init__(self, model, preprocess, data_path):

        self.img_path = data_path + "/images/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.preprocess = preprocess
        with open(data_path + "matched_Corpus2.csv", 'r') as inf:
            self.data = pd.read_csv(inf, header=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx][18]
        label = self.data.iloc[idx][14]
        claim = self.data.iloc[idx][13]
        evidence = self.data.iloc[idx][16]
        if label == "NEI":
            label = 0
        elif label == "supported":
            label = 1
        elif label == "refuted":
            label = 2

        claim_inp = clip.tokenize(claim, truncate=True).squeeze().to(self.device)
        evidence_inp = clip.tokenize(evidence, truncate=True).squeeze().to(self.device)
        img_inp = self.preprocess(Image.open(self.img_path + img)).to(self.device)
        label_inp = torch.as_tensor(int(label)).to(self.device, torch.long)

        return claim_inp, evidence_inp, img_inp, label_inp

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.cross = nn.Linear(feature_dim, 3)
    def forward(self, a, b):
        # 生成查询、键和值
        query = self.query(a)
        key = self.key(b)
        value = self.value(b)
        # print(key.size())
        # 计算注意力分数

        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores, dim=-1)

        # 应用注意力分数
        attn_output = torch.matmul(attn_scores, value)
        attn_output = self.cross(attn_output)

        return attn_output, attn_scores

class Adapter(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(Adapter, self).__init__()
        self.fc_cat = nn.Linear(512 * 3, num_classes)
        self.fc_claim = nn.Linear(512, num_classes)
        self.fc_img = nn.Linear(512, num_classes)
        self.fc_evidence = nn.Linear(512,num_classes)
        self.cross_attention = CrossAttention(512)

        self.fc_meta = nn.Linear(num_classes * 6, num_classes)


    def forward(self, claim, evidence, img, fused):
        # 获取文本和图像的输出
        # claim_out = self.fc_claim(claim)
        # evidence_out = self.fc_evidence(evidence)
        # img_out = self.fc_img(img)
        # fused_out = self.fc_cat(fused)

        # 应用交叉注意力机制
        attn_ce, _ = self.cross_attention(claim, evidence)
        attn_ec, _ = self.cross_attention(evidence, claim)
        attn_ci, _ = self.cross_attention(claim, img)
        attn_ic, _ = self.cross_attention(img, claim)
        attn_ei, _ = self.cross_attention(evidence, img)
        attn_ie, _ = self.cross_attention(img, evidence)
        combined_out = torch.cat((attn_ce, attn_ec, attn_ci, attn_ic, attn_ei, attn_ie), dim=-1)
        # 合并来自基学习器的输出
        # combined_out = torch.cat((claim_out, evidence_out, img_out, fused_out, attn_ce, attn_ec, attn_ci, attn_ic, attn_ei, attn_ie), dim=1)
        meta_out = self.fc_meta(combined_out)

        return meta_out


# a = Mocheg_Dataset(1, 2, "../mocheg/val/matched_Corpus2.csv", "../mocheg/val/images/")

