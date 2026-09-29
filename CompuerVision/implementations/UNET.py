import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

NUM_CLASSES = 21      # 20 VOC classes + background
IGNORE_INDEX = 255    # VOC marks object borders with 255
IMG_SIZE = 256        # must be divisible by 16 (4 poolings)

train_joint = v2.Compose([
    v2.Resize([IMG_SIZE, IMG_SIZE]),          # bilinear for the image, nearest for the mask
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),    # scales the image only; masks are left untouched
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_joint = v2.Compose([
    v2.Resize([IMG_SIZE, IMG_SIZE]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_transform(joint):
    def transform(img, mask):
        img = tv_tensors.Image(F.pil_to_tensor(img))
        mask = tv_tensors.Mask(F.pil_to_tensor(mask))     # palette indices, (1, H, W)
        img, mask = joint(img, mask)
        return img, mask.squeeze(0).long()                # (3,H,W), (H,W)
    return transform


dataset = torchvision.datasets.VOCSegmentation(
    root="CompuerVision/implementations/data", year="2012", image_set="train",
    download=True, transforms=make_transform(train_joint),
)
dataset_test = torchvision.datasets.VOCSegmentation(
    root="CompuerVision/implementations/data", year="2012", image_set="val",
    download=True, transforms=make_transform(test_joint),
)

dataloader_train = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=2)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # in_ch = in_ch//2 (upsampled) + in_ch//2 (skip)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)   # bottleneck
        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up3 = Up(base * 4, base * 2)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)   # logits (N, num_classes, H, W)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (pixel accuracy, mean IoU over classes present), ignoring 255."""
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(1)
        valid = masks != IGNORE_INDEX
        idx = masks[valid] * NUM_CLASSES + preds[valid]
        confusion += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    tp = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = tp / union.clamp(min=1)
    present = confusion.sum(1) > 0
    return (tp.sum() / confusion.sum()).item(), iou[present].mean().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total = 0.0
        for imgs, masks in dataloader_train:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        acc, miou = evaluate(model, dataloader_test, device)
        print(f"epoch {epoch}: loss {total / len(dataloader_train):.4f} | val acc {acc:.3f} | val mIoU {miou:.3f}")

    torch.save(model.state_dict(), "CompuerVision/implementations/unet.pt")
