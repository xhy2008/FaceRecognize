import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from mobilefacenet import MobileFacenet, ArcMarginProduct


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        identity_dirs = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d))])
        
        self.label_to_idx = {label: idx for idx, label in enumerate(identity_dirs)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        for identity in identity_dirs:
            identity_path = os.path.join(root_dir, identity)
            for img_name in os.listdir(identity_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(identity_path, img_name))
                    self.labels.append(self.label_to_idx[identity])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=112):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'net_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['net_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded pretrained weights from {checkpoint_path}")
    return model


def train_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = 112
    num_classes = 395
    embedding_size = 256
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 5e-4
    
    data_dir = './Cleaned_CASIA_FaceV5'
    pretrained_path = './mobilefacenet.ckpt'
    save_path = './mobilefacenet_finetuned.ckpt'
    
    train_transform, _ = get_transforms(img_size)
    
    train_dataset = FaceDataset(data_dir, transform=train_transform)
    num_classes = len(train_dataset.label_to_idx)
    print(f"Number of identities: {num_classes}")
    print(f"Number of images: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    backbone = MobileFacenet()
    backbone = load_pretrained_model(backbone, pretrained_path, device)
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    for param in backbone.linear1.parameters():
        param.requires_grad = True
    for param in backbone.conv2.parameters():
        param.requires_grad = True
    for param in backbone.linear7.parameters():
        param.requires_grad = True
    
    metric_fc = ArcMarginProduct(
        in_features=embedding_size,
        out_features=num_classes,
        s=25.0,
        m=0.3,
        easy_margin=False
    )
    
    params = []
    for param in backbone.linear1.parameters():
        params.append(param)
    for param in backbone.conv2.parameters():
        params.append(param)
    for param in backbone.linear7.parameters():
        params.append(param)
    for param in metric_fc.parameters():
        params.append(param)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    backbone = backbone.to(device)
    metric_fc = metric_fc.to(device)
    
    print("Starting Finetune...")
    
    for epoch in range(num_epochs):
        backbone.train()
        metric_fc.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            embeddings = backbone(images)
            outputs = metric_fc(embeddings, labels)
            
            all_preds.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        max_probs, pred_labels = torch.max(all_preds, dim=1)
        avg_confidence = max_probs.mean().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Avg Conf: {avg_confidence:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'metric_fc_state_dict': metric_fc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            print(f"Model saved to {save_path}")
    
    final_checkpoint = {
        'backbone_state_dict': backbone.state_dict(),
        'metric_fc_state_dict': metric_fc.state_dict(),
    }
    torch.save(final_checkpoint, save_path)
    print(f"Finetuning completed! Final model saved to {save_path}")


if __name__ == "__main__":
    train_finetune()
