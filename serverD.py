from pyopp import cSimpleModule, cMessage
import sys 
from collections import defaultdict


class Server(cSimpleModule):
    
    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)

    
    def handleMessage(self, msg):

        if msg.getName() == "Performance":            
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
            self.delete(msg)
    
    def finish(self):
        for round in self.hit_ratios.keys():
            if(len(self.hit_ratios[round]) == self.num_participants): # if we receive all the users' model we evaluate them and then compute an average perf
                print("round = ", round)
                avg = sum(self.hit_ratios[round]) / self.num_participants
                print("Average Test HR = ",avg)
                avg = sum(self.ndcgs[round]) / self.num_participants
                print("Average Test NDCG = ",avg)
                sys.stdout.flush()
                
