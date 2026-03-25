import torch
import torchvision
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
class ViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.encoder = vit.encoder
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.hidden_dim = vit.hidden_dim
        del vit
        
    def _process_input(self, x):
        n, c, h, w = x.shape
        p = 16
        n_h = h // p
        n_w = w // p
        
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x
    
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]
        
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = self.encoder(x)
        
        return x[:, 0]

def train_epoch(model, classifier, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=device.type == 'cuda'):
            features = model(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, classifier, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                features = model(images)
                logits = classifier(features)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    return total_loss / len(loader), 100. * correct / total

if __name__ == '__main__':
    base_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset=torchvision.datasets.ImageFolder(root='./Cleaned_CASIA_FaceV5', transform=train_transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.transform = train_transform
    val_dataset.transform = base_transform
    train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
    vit=ViTFeatureExtractor(pretrained=True)

    for param in vit.encoder.layers[:-1].parameters():
        param.requires_grad = False#冻结encoder层
    for param in vit.conv_proj.parameters():
        param.requires_grad = False#冻结卷积投影层

    trainable_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vit.parameters())
    print(f"ViT 总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"训练集数量: {len(train_dataset)}")
    print(f"验证集数量: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = vit.to(device)
    classifier=nn.Linear(vit.hidden_dim, len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, list(vit.parameters()) + list(classifier.parameters())), lr=3e-4, weight_decay=5e-4)

    scaler = GradScaler()

    num_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0

    print("开始训练...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(vit, classifier, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(vit, classifier, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'vit_state_dict': vit.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_vit_model.pth')
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    torch.save({
        'vit_state_dict': vit.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, 'final_model.pth')
    print("训练完成！")
