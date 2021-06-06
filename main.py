import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%%
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("running on the GPU...\n")
else:
    device = torch.device('cpu')
    print("running on the CPU...\n")

#%%
data = pd.read_csv('HandInfo.csv')

labels = data[['gender', 'aspectOfHand']].agg('-'.join, axis=1)
labels_df = pd.DataFrame(labels, columns=['label'])
labels_array = labels_df.iloc[:,0].values
le = LabelEncoder()
labels_array = le.fit_transform(labels_array)
labels_df = pd.DataFrame(labels_array, columns=['label'])

img_file_names = os.listdir('Hands')
img_paths = []
for img_file_name in img_file_names:
    img_path = os.path.join('Hands', img_file_name)
    img_paths.append(img_path)

img_paths_df = pd.DataFrame(img_paths, columns=['img_path'])

final_data = pd.concat( [labels_df, img_paths_df], axis = 1)

train_data, test_data = train_test_split(final_data, test_size=0.15, random_state=0)
train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=0)

train_data_labels = train_data.iloc[:,0]
train_data_images = train_data.iloc[:,1]

test_data_labels = test_data.iloc[:,0]
test_data_images = test_data.iloc[:,1]

validation_data_labels = validation_data.iloc[:,0]
validation_data_images = validation_data.iloc[:,1]


#%%
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(128),
     transforms.ToTensor(),
     transforms.Normalize([0.458, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
#%%
class HandDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = cv2.imread(self.X.iloc[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image)
            
        if self.y is not None:
            return (image, self.y.iloc[i])
        else:
            return image
        
# datasets
train_dataset = HandDataset(train_data_images, train_data_labels, transform)
validation_dataset = HandDataset(validation_data_images, validation_data_labels, transform)
test_dataset = HandDataset(test_data_images, test_data_labels, transform) 

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


#%%

# define the neural net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=13056, out_features=256)
        self.dropout_layer = nn.Dropout(0.5)                                 
        self.fc2 = nn.Linear(in_features=256, out_features=8)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)                                             
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)                                             
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
net = Net().to(device) 
print(net)

#%%
# loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9)

#%%

def train(net, train_loader):
    print('\nStart Training...\n')
    
    loss_list = []
    acc_list = []
    loss_valid_list = []
    acc_valid_list = []
    
    for epoch in range(10): 
        running_loss = 0
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        running_loss_valid = 0
        with torch.no_grad():
            for data_valid in validation_loader:
                inputs_valid, labels_valid = data_valid[0].to(device), data_valid[1].to(device) 
                outputs_valid = net(inputs_valid)
                loss_valid = criterion(outputs_valid, labels_valid)
                running_loss_valid += loss_valid.item()
            
        epoch_no = epoch + 1
        
        loss = running_loss/len(train_loader)
        acc, conf_m = test(net, train_loader)
        loss_valid = running_loss_valid/len(validation_loader)
        acc_valid, conf_m_valid = test(net, validation_loader)
        
        loss_list.append(loss)
        acc_list.append(acc)
        loss_valid_list.append(loss_valid)
        acc_valid_list.append(acc_valid)
        
        print('[Epoch %d]' %(epoch_no))
        print('training loss: %.5f' %(loss))
        print('training accuracy: %0.5f %%' % (acc))
        print('validation loss: %.5f' %(loss_valid))
        print('validation accuracy: %0.5f %%' % (acc_valid))
        print('\n')
        
    print('Done Training.\n')
    return loss_list, acc_list, loss_valid_list, acc_valid_list
 

  
    
def test(net, data_loader):
    
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(8, 8)
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
    return (100 * correct / total), confusion_matrix
    
    
loss_list, acc_list, loss_valid_list, acc_valid_list = train(net, train_loader)

epoch = [1,2,3,4,5,6,7,8,9,10]

plt.plot( epoch, loss_list, marker='', color='skyblue', linewidth=2, label = 'train loss')
plt.plot( epoch, loss_valid_list,  marker='', color='darkblue', linewidth=2, label = 'validation loss')
plt.xlabel("epoch")
plt.ylabel("value of loss")
plt.legend()
plt.show()

plt.plot( epoch, acc_list, marker='', color='skyblue', linewidth=2, label = 'train accuracy')
plt.plot( epoch, acc_valid_list,  marker='', color='darkblue', linewidth=2, label = 'validation accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.legend()
plt.show()


test_accuracy, conf_m = test(net, test_loader)
print('Confusion Matrix:')
print(conf_m)
print('Accuracy of the network on test images: %0.5f %%' % (test_accuracy))






