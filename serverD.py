from pyopp import cSimpleModule, cMessage
import sys 
from collections import defaultdict


class Server(cSimpleModule):
    
    def initialize(self):
        self.init_round = 790
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)

    
    def handleMessage(self, msg):

        if msg.getName() == "Model":            
            self.hit_ratios[msg.round].append(msg.hit_ratio)
            self.ndcgs[msg.round].append(msg.ndcg)
            print("init round",self.init_round)

           
            if(len(self.hit_ratios[self.init_round]) == self.num_participants): # if we receive all the users' model we evaluate them and then compute an average perf
                avg = sum(self.hit_ratios[self.init_round]) / self.num_participants
                print("Average Test HR = ",avg)
                avg = sum(self.ndcgs[self.init_round]) / self.num_participants
                print("Average Test NDCG = ",avg)
                sys.stdout.flush()
                self.hit_ratios[self.init_round] = [] 
                self.ndcgs[self.init_round] = []
                self.init_round -= 10
            
        

        self.delete(msg)