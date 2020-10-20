#Assignment#2
#2018707017 정용훈

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2= nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential(
            nn.Linear(128*4*4, 120),
            nn.Linear(120, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out=self.layer3(out)
        out = F.dropout(out, training=self.training)
        out=out.reshape(out.size(0),-1)
        out=self.fc_layer(out)
        return out


device = 'cpu'

num_epochs=30
num_classes=10
batch_size=64
learning_rate=0.001

transform = transforms.Compose(
    [

        transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


train_dataset=torchvision.datasets.CIFAR10(root='./DOWN/',
                                         train=True,
                                         transform=transform,
                                         download=True)

test_dataset=torchvision.datasets.CIFAR10(root='./DOWN/',
                                         train=False,
                                         transform=transform)

#Data loader
train_loader= torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader= torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = ConvNet(num_classes).to(device)

#Loss and Optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(ConvNet)

#Train the model
total_step=len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)

        #Forward pass
        outputs=model(images)
        loss=criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%100==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


#Save the model checkpoint
torch.save(model.state_dict(), 'CIFAR10_model.skpt')

#Test the model
model.eval()
with torch.no_grad():
    correct =0
    total=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _, predicted=torch.max(outputs.data, 1)
        total += labels.size(0)
        correct+=(predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images : {} %'.format(100*correct/total))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))