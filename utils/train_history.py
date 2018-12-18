import pickle
import torch
class train_history:
    def __init__(self,names):
        self.history = {}
        self.one_batch_history={}
        self.names = names
        for name in names:
            self.history[name] = []
            self.one_batch_history[name] = []
            
    def add_params(self,params):
        assert len(params) == len(self.names)
        for i,item in enumerate(params):
            self.one_batch_history[self.names[i]].append(item)
        
    def check_current_avg(self):
        for name in self.names:
            temp = {}
            for name in self.names:
                loss = torch.mean(torch.FloatTensor(self.one_batch_history[name]))
                temp[name] = loss
                self.history[name].append(loss)
                self.one_batch_history[name].clear()
            return temp
        
    def get_last_param_str(self):
        result_string = ''
        for name in self.names:
            result_string += str.format('{}:{:.3f},',name,self.history[name][-1])
        return result_string
    
    def save_train(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)
    
    def load_train(self,path):
        with open(path, 'rb') as file:
            self.history = pickle.load(file)