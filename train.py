

import json
import numpy as np
import time as tm

from nltk_utls import tokenize, stem, bag_of_words  # all required function for nltk is created in this file 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

def info():
     syn = f"Training ~COMPLETED~ \n The data is loaded in {FILE}\n"
     for i in syn:
          print(i,end="")
          tm.sleep(0.035)
          

with open('intents.json', 'r') as f:
     intents = json.load(f)
     
all_words = []
tags = []
xy = []

# bring the required elements in list
for intent in intents['intents']:
     tag = intent['tag']
     tags.append(tag)
     for pattern in intent['patterns']:
          w = tokenize(pattern)
          all_words.extend(w)
          xy.append((w,tag))
     
ignore_words = ['?','!',',','.']

# Bring the complicated text to its stem state
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
X_train = []
y_train = []


for (tokenised_sentence, tag) in xy:
     bag  = bag_of_words(tokenised_sentence, all_words)
     X_train.append(bag)
     
     label = tags.index(tag)                # indexing the tags
     y_train.append(label)
     
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
     def __init__(self):
          self.n_samples = len(X_train)
          self.x_data = X_train
          self.y_data = y_train
  
     def __getitem__(self, index):
          return self.x_data[index], self.y_data[index]
     
     def __len__(self):
          return self.n_samples

# creating hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size =len(X_train[0])
learning_rate = 0.001
num_epochs = 2000


     
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_epochs):
     for (words, labels) in train_loader:
          words = words.to(device)
          labels = labels.to(device)
          
          #forward training
          outputs = model(words)
          loss = criterion(outputs, labels)
          
          # backword and optimizer step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
     if (i + 1) % 100 == 0:
          print(f'epoch {i + 1}/{num_epochs}, loss = {loss.item():.4f}')
          
print(f'final loss ,loss = {loss.item():.4f}')

          
data = {
     "model_state" : model.state_dict(),
     "input_size" : input_size,
     "output_size" : output_size,
     "hidden_size" : hidden_size,
     "all_words" : all_words,
     "tags" : tags
}

FILE = "data.pth"
torch.save(data,FILE)

info()

          



     





          

