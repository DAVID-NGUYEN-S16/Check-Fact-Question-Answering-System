
from torch import nn

from tqdm.notebook import tqdm
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import  RobertaModel
from transformers import AutoModel
from huggingface_hub import login
device = "cpu"


class ClaimVerification(nn.Module):
    def __init__(self, n_classes, name_model):
        super(ClaimVerification, self).__init__()
        self.bert = AutoModel.from_pretrained(name_model)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

class Rational_Tagging(nn.Module):
    def __init__(self,  hidden_size):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_1 = self.W1(h_t)
        h_1 = F.relu(h_1)
        p = self.w2(h_1)
        p = torch.sigmoid(p)
        return p
    
class RTLoss(nn.Module):
    
    def __init__(self, hidden_size = 768, device = 'cuda'):
        super(RTLoss, self).__init__()
        self.device = device
    
    def forward(self, pt: torch.Tensor, Tagging:  torch.Tensor):
        '''
        Tagging: list paragraphs contain value token. If token of the paragraphas is rationale will labeled 1 and other will be labeled 0 
        
        RT: 
                    p^r_t = sigmoid(w_2*RELU(W_1.h_t))
            
            With:
                    p^r_t constant
                    w_2 (d x 1)
                    W_1 (d x d)
                    h_t (1 x d)
                    
            This formular is compute to each token in paraphase. I has convert into each paraphase
            
                    p^r_t = sigmoid(w_2*RELU(W_1.h))
                    
                    With:
                            p^r (1 x n) with is number of paraphase
                            w_2 (d x 1)
                            W_1 (d x d)
                            h (n x d) 
                            
        '''
        
        Tagging = torch.tensor(Tagging, dtype=torch.float32).to(device)
                
        total_loss = torch.tensor(0, dtype= torch.float32).to(device)
        
        N = pt.shape[0]
                
        for i, text in enumerate(pt):
            T = len(Tagging[i])
            Lrti = -(1/T) * (Tagging[i]@torch.log(text) + (1.0 - Tagging[i]) @ torch.log(1.0 - text) )[0]
            total_loss += Lrti
            
        return total_loss/N

class BaseLoss(nn.Module):
    
    def __init__(self):
        super(BaseLoss, self).__init__()
    
    def forward(self, start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions:torch.Tensor , end_positions:torch.Tensor ):
        
        batch_size = start_logits.shape[0]
        
        start_zero = torch.zeros(start_logits.shape).to('cuda:0')
        end_zero = torch.zeros(start_logits.shape).to('cuda:0')

        for batch, y in enumerate(start_positions):
            start_zero[batch][y][0] = 1
            
        for batch, y in enumerate(end_positions):
            end_zero[batch][y][0] = 1

        start_logits = F.softmax(start_logits, dim=1)
        end_logits = F.softmax(end_logits, dim=1)
        
        proba_start = (start_logits*start_zero).sum(dim=1) 
        proba_end = (end_logits*end_zero).sum(dim=1) 

        proba_start = torch.log(proba_start )
        proba_end = torch.log(proba_end )

        total_loss = -(1/(2*batch_size)) * torch.sum(proba_start + proba_end)
        
        return total_loss
class comboLoss(nn.Module):
    def __init__(self, alpha: int = 1, beta: int = 1):
        
        super(comboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.BaseLoss = BaseLoss()
        self.RTLoss = RTLoss()
        
    def forward(self, output: dict):

        start_logits = output['start_logits']
        end_logits = output['end_logits']
        
        start_positions = output['start_positions']
        end_positions = output['end_positions']
        
        Tagging = output['Tagging']
        pt = output['pt']
        
        loss_base = self.BaseLoss(start_logits = start_logits, end_logits = end_logits, start_positions = start_positions, end_positions = end_positions)
        retation_tagg_loss  = self.RTLoss(pt = pt, Tagging = Tagging)
        
        total_loss = self.alpha*loss_base + self.beta*retation_tagg_loss
        
        return total_loss
# Code model baseline 
class ModelQA(nn.Module):
    def __init__(self, name_model):
        super(ModelQA, self).__init__()
        
#         self.number_label = config.number_labels
        
        self.model = RobertaModel.from_pretrained(name_model)
        self.config = self.model.config
        # Use FC layer to get the start logits l_start and end logits_end
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, 2)
        
        self.tagging = Rational_Tagging(self.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        '''
        output: model will return hidden state, pooler,..
        
        qa_output: return (batch, row, colum) example (1, 8, 768)
        
        logits contain probability of an word is start position and end position
        
        example:
                    tensor([[[-0.1880, -0.0796],
                            [-0.2347, -0.1440],
                            [-0.2825, -0.1179],
                            [-0.3406, -0.1836],
                            [-0.3912,  0.0133],
                            [-0.1169, -0.3032],
                            [-0.3016, -0.1336],
                            [-0.1779, -0.0750]]], grad_fn=<ViewBackward0>) 
        
        '''
        output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        
        qa_ouputs = output[0]
        
        logits = self.qa_outputs(qa_ouputs)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        pt =  self.tagging(qa_ouputs)
        
        return pt, start_logits, end_logits