
from ast import Pass
import random
import json
from turtle import clear
import time as tm
import torch
import random 
from model import NeuralNet
from nltk_utls import bag_of_words, tokenize


def aniprint(a):
     for i in a:
          print(i,end="")
          tm.sleep(random.uniform(0.001, 0.02))
     


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
     intents = json.load(f)
     
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size,hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = ""
aniprint("{ type 'qt' to EXIT }\n \nHello..!!, \nI am juan \nYour new bot\nLets chat..!! ")

while True:
     sentence = input("\n==>> ")
     if sentence == "qt":
          break
     
     
     sentence = tokenize(sentence)
     X = bag_of_words(sentence, all_words)
     X = X.reshape(1,X.shape[0])
     X = torch.from_numpy(X)
     
     output = model(X)
     _, predicted = torch.max(output,dim=1)
     tag = tags[predicted.item()]
     
     probs = torch.softmax(output, dim = 1)
     prob = probs[0][predicted.item()]
     
     if prob.item() > 0.75:
          for intent in intents["intents"]:
               if tag == intent["tag"]:
                    aniprint((str(bot_name))+ ": " + str(random.choice(intent["responses"])))
     else:
          print(f"{bot_name} : \n x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x\n   i am sorry its out of my understanding for now.. \n Hold tight..!! \n Our developers are working hard to bring this feature..\n x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x \n ")
          
          