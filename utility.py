import os
from configparser import ConfigParser

####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import random
import numpy as np
import torch
from torch import nn

cost_file_name ="overheadAlpha08.ini"


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   np.random.seed(1)
   random.seed(1)
   torch.manual_seed(1)

def init_cost_file():
    config_object = ConfigParser()
    config_object["performance"] = {
    "Aggregation_Time_Per_Round" : 0,
    "Aggregation_Time_Total": 0,
    "Transfer_Time_Per_Round_To_Server": 0,
    "Transfer_Time_Total_To_Server": 0,
    "Transfer_Time_Init_Total": 0
    }
    with open(cost_file_name,"w") as cost_file:
        config_object.write(cost_file)

reset_random_seeds()
init_cost_file()


class Network(nn.Module):
        def __init__(self,num_users, num_items,latent_dim = 8):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.latent_dim = latent_dim
            self.sigmoid = nn.Sigmoid()
            self.loss_func = nn.BCELoss()
            self.users_embeddings = nn.Embedding(self.num_users, self.latent_dim, dtype= torch.float32)
            # nn.init.normal_(self.users_embeddings.weight)
            self.items_embeddings = nn.Embedding(self.num_items, self.latent_dim, dtype= torch.float32)
            # nn.init.normal_(self.items_embeddings.weight)
            self.ll = nn.Linear(self.latent_dim, 1)
            # Define sigmoid activation and softmax output 
                
        def forward(self, users_input, items_input):
            # Pass the input tensor through each of our operations
            user_embedding = self.users_embeddings(users_input)
            item_embedding = self.items_embeddings(items_input)

            # flattening the embeddings
            user_embedding = user_embedding.view(user_embedding.size(0), -1)
            item_embedding = item_embedding.view(item_embedding.size(0), -1)

            element_wise_product = torch.mul(user_embedding,item_embedding)
            output = self.ll(element_wise_product)
            output = self.sigmoid(output)
            return output.squeeze(1)



def get_model(num_users,num_items):
    model = Network(num_users, num_items)    
    return model



def save_per_round_cost(time_t, property):
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    performance[property] = str(time_t)
    with open(cost_file_name,"w") as cost_file:
        config_object.write(cost_file)

def save_whole_cost(time_t, property):
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    if property in performance:
        performance[property] = str( float(performance[property]) + time_t)
    else:
        performance[property] = str(time_t)
    with open(cost_file_name,"w") as cost_file:
        config_object.write(cost_file)