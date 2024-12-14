import torch
import torchvision.transforms as transform
import os
from torch import optim
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy


class convblock(torch.nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.c1 = torch.nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.c2 = torch.nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.c1(x)
        x1 = self.relu1(x1)
        x2 = self.c2(x1)
        x2 = self.relu2(x2)
        return x2
    
class convdown(torch.nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.c = convblock(inchannel,outchannel)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
    
    def forward(self,x):
        x1 = self.c(x)
        x2 = self.maxpool(x1)
        return x1, x2

class convup(torch.nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(inchannel, inchannel//2, kernel_size=2, stride=2)
        self.c = convblock(inchannel, outchannel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        xs =  self.c(x)
        return xs

class actualunet(torch.nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.down1 = convdown(inchannel,64)
        self.down2 = convdown(64,128)
        self.down3 = convdown(128,256)
        self.down4 = convdown(256,512)
        
        self.bottleneck = convblock(512,1024)

        self.up1 = convup(1024,512)
        self.up2 = convup(512,256)
        self.up3 = convup(256,128)
        self.up4 = convup(128,64)

        self.out = torch.nn.Conv2d(64,outchannel,kernel_size=1)


    def forward(self,x):
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(y1)
        x3, y3 = self.down3(y2)
        x4, y4 = self.down4(y3)

        b = self.bottleneck(y4)

        z1 = self.up1(b,x4)
        z2 = self.up2(z1,x3)
        z3 = self.up3(z2,x2)
        z4 = self.up4(z3,x1)

        out = self.out(z4)
        return out


transforms = transform.Compose([
    transform.Resize((512,512)),
    transform.ToTensor()
])

class blooddataset(dataset.Dataset):
    def __init__(self,image,mask,transform):
        self.image = image
        self.mask = mask
        self.images = []
        for i in os.listdir(self.image):
            self.images.append(self.image + i)
        
        self.masks = []
        for i in os.listdir(self.mask):
            self.masks.append(self.mask + i)
        
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

        self.transform = transform

    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

imgpath = "/kaggle/input/bccd-dataset-with-mask/BCCD Dataset with mask/train/original/"
maskpath = "/kaggle/input/bccd-dataset-with-mask/BCCD Dataset with mask/train/mask/"
dataset = blooddataset(imgpath,maskpath,transform=transforms)
dataloader = DataLoader(dataset=dataset,batch_size=8,shuffle=True,pin_memory=False,num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = actualunet(3,1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
lossf = torch.nn.BCEWithLogitsLoss()

torch.cuda.empty_cache()

for epoch in range(2):
    model.train()
    for imgnmask in dataloader:
        img = imgnmask[0].float().to(device)
        mask = imgnmask[1].float().to(device)

        predict = model(img)
        optimizer.zero_grad()

        loss = lossf(predict, mask)

        loss.backward()
        optimizer.step()
    print(epoch + 1)

torch.save(model.state_dict(), 'checkpoint.pth')

imagepth = "/kaggle/input/bccd-dataset-with-mask/BCCD Dataset with mask/test/original/e02301ac-68f7-4daf-b2ba-254b917f1cca.jpg"
image = Image.open(imagepth).convert("RGB")

imaget = transforms(image).unsqueeze(0).to(device)
result = model(imaget)

result = result.squeeze(0)
result = torch.sigmoid(result).cpu().detach().numpy()
result = (result > 0.5).astype(float)

plt.figure(figsize=(15, 16))
plt.subplot(121), plt.imshow(imaget.squeeze().cpu().detach().numpy().transpose(1, 2, 0)), plt.title("Image")
plt.subplot(122), plt.imshow(result[0], cmap="gray"), plt.title("Result")
plt.show()

original_size = image.size

result_image = Image.fromarray(result[0] * 255).convert('L')
result_resized = result_image.resize(original_size, Image.BILINEAR) 
result_resized = numpy.array(result_resized)
result_resized = (result_resized > 0.5).astype(float)
result_resized = Image.fromarray((result_resized * 255).astype(numpy.uint8)).convert('L')

save_image_path = '/kaggle/working/'
os.makedirs(save_image_path, exist_ok=True)

image.save(os.path.join(save_image_path, 'original_image.jpg'))
result_resized.save(os.path.join(save_image_path, 'predicted_mask.png'))