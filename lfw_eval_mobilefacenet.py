import torchvision.transforms as transforms
import torch
from mobilefacenet import MobileFacenet
from PIL import Image
import os
from tqdm import tqdm
import pickle
print("loading model")
model = MobileFacenet().to("cuda")
model.load_state_dict(torch.load("mobilefacenet_finetuned.ckpt")["backbone_state_dict"])
model.eval()
print("loading dataset")
same_pairs=[]
diff_pairs=[]
with open("lfw/pairs.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 处理每一行数据
        parts = line.split()
        if len(parts)==3:
            same_pairs.append((parts[0],int(parts[1]),int(parts[2])))
        elif len(parts)==4:
            diff_pairs.append((parts[0],int(parts[1]),parts[2],int(parts[3])))
    del lines
print(f"same pairs: {len(same_pairs)}")
print(f"diff pairs: {len(diff_pairs)}")
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
same_sims=[]
diff_sims=[]
print("eval same pairs")
for pair in tqdm(same_pairs):
    person,img1,img2=pair
    img1 = Image.open(os.path.join("lfw/images",person,os.listdir(os.path.join("lfw/images",person))[img1-1]))
    img2 = Image.open(os.path.join("lfw/images",person,os.listdir(os.path.join("lfw/images",person))[img2-1]))
    img1 = transform(img1).unsqueeze(0).to("cuda")
    img2 = transform(img2).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feat1 = model(img1)
        feat2 = model(img2)
    sim = torch.cosine_similarity(feat1,feat2,dim=1).item()
    same_sims.append(sim)
print(f"min same sim: {min(same_sims)}")
print(f"avg same sim: {sum(same_sims)/len(same_sims)}")
print("eval diff pairs")
for pair in tqdm(diff_pairs):
    person,img1,person2,img2=pair
    img1 = Image.open(os.path.join("lfw/images",person,os.listdir(os.path.join("lfw/images",person))[img1-1]))
    img2 = Image.open(os.path.join("lfw/images",person2,os.listdir(os.path.join("lfw/images",person2))[img2-1]))
    img1 = transform(img1).unsqueeze(0).to("cuda")
    img2 = transform(img2).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feat1 = model(img1)
        feat2 = model(img2)
    sim = torch.cosine_similarity(feat1,feat2,dim=1).item()
    diff_sims.append(sim)
print(f"max diff sim: {max(diff_sims)}")
print(f"avg diff sim: {sum(diff_sims)/len(diff_sims)}")
with open("lfw_mobilefacenet_finetuned_sims.pkl", "wb") as f:
    pickle.dump({"same_sims": same_sims, "diff_sims": diff_sims}, f)