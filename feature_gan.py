import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets,transforms
#import matplotlib.pyplot as plt

#这个是一个简单的生成对抗网络，其实是一个特征迁移网络，是将market1501在网络提取后的特征，迁移到duke经过网络提取特征后的空间里，寻思能对跨域REID有帮助，但是实验结果并不好。


class feature_extract_net(nn.Module):
    def __init__(self):
        super(feature_extract_net,self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load('./model/resnet50.pth')) #为了快速加载resnet模型，而不是每次都需要下载这个模型的参数，博客讲解地址：https://blog.csdn.net/t20134297/article/details/103885879
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        return x

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),

            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),

            nn.ReLU(),
        )

    def forward(self,x):
        x = self.gen(x)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024,256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            # nn.Linear(256,128),
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(128),

            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.dis(x)
        return x

#loss_f = torch.nn.BCEWithLogitsLoss()
# loss_f = torch.nn.BCELoss()
adversarial_loss = torch.nn.BCELoss()

transform_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

trans = transforms.Compose(transform_list)

image_dataset1 = datasets.ImageFolder('./Market/pytorch/train', trans)
image_dataset2 = datasets.ImageFolder('./duke/pytorch/train', trans)
try_dataset1 = datasets.ImageFolder('./try_data1', trans)
try_dataset2 = datasets.ImageFolder('./try_data2', trans)


dataloader1 = torch.utils.data.DataLoader(image_dataset1, batch_size=16, shuffle=True)
dataloader2 = torch.utils.data.DataLoader(image_dataset2, batch_size=16, shuffle=True)
try_dataloader1 = torch.utils.data.DataLoader(try_dataset1, batch_size=2)
try_dataloader2 = torch.utils.data.DataLoader(try_dataset2, batch_size=2)

net = feature_extract_net()
net.cuda()
net.eval()

gen = generator()
gen.cuda()


dis = discriminator()
dis.cuda()

optimizer_dis = torch.optim.Adam(dis.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_gen = torch.optim.Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.999))

batches_track = []
g_loss_list = []
d_loss_list = []
batches_done = 0

epoch_n=1
for epoch in range(epoch_n):
    for  data_group in zip(dataloader1,dataloader2):
        data1 = data_group[0]
        data2 = data_group[1]

        inputs1,labels1 = data1
        inputs2,labels2 = data2

        inputs1 = inputs1.cuda()
        inputs2 = inputs2.cuda()

        batch_s1 = inputs1.size()[0]
        batch_s2 = inputs2.size()[0]
        if batch_s1 != batch_s2:
            break

        inputs1 = net(inputs1)
        inputs2 = net(inputs2)

        valid = torch.Tensor(inputs1.shape[0],1).fill_(1.0)
        valid = valid.cuda()
        fake = torch.Tensor(inputs1.shape[0],1).fill_(0.0)
        fake = fake.cuda()

        fea_gen = gen(inputs1)


        gen_pre = dis(fea_gen)
        real_pre = dis(inputs2)

        g_loss = adversarial_loss(gen_pre,valid)

        optimizer_gen.zero_grad()
        g_loss.backward()
        optimizer_gen.step()

        optimizer_dis.zero_grad()
        real_loss = adversarial_loss( dis(inputs2) ,valid)
        fake_loss = adversarial_loss(dis(fea_gen.detach()),fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_dis.step()

        batches_done = batches_done + 1
        batches_track.append(batches_done)
        g_loss_list.append(g_loss.item())
        d_loss_list.append(d_loss.item())

        print('Epoch {}/{},batches_done {}'.format(epoch+1,epoch_n,batches_done))
        print("discriminator Loss:{:.4f}".format(d_loss.item()))
        print('generator loss:{:.4f}..'.format(g_loss.item()))

torch.save(gen.cpu().state_dict(), 'try_net.pth')

# fig,ax = plt.subplots()
# ax.plot(batches_track,d_loss_list,label='discriminator loss')
# ax.plot(batches_track,g_loss_list,label='generator loss')
# ax.legend()
# plt.show()





