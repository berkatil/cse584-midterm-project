from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DualEncoder(torch.nn.Module):
    def __init__(self,
                max_xi_length,
                max_xj_length,
                number_of_classification_layers,
                sbert_or_luar):
        super(DualEncoder, self).__init__()
        self.max_xi_length = max_xi_length
        self.max_xj_length = max_xj_length

        if sbert_or_luar == "sbert":
            self.model_i = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
            self.model_j = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
            hidden_size = self.model_i[1].word_embedding_dimension
            self.tokenizer = None
        elif sbert_or_luar == "luar":
            self.model_i = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)
            self.model_j = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)
            hidden_size = self.model_i.config.embedding_size
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")
        else:
            raise ValueError("Unsupoorrted model type", sbert_or_luar)
    
        if number_of_classification_layers == 1:
            self.dense = torch.nn.Linear(hidden_size*2,7)
        else: #IT HAS TO BE 2
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(hidden_size*2,hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size,7),
            )
        self.dense = self.dense.to(self.model_i.device)
        self.sbert_or_luar = sbert_or_luar

    def forward(self, xi, xj):
        if self.sbert_or_luar == "sbert":
            embeddings_i = torch.from_numpy(self.model_i.encode(xi))
            embeddings_j = torch.from_numpy(self.model_j.encode(xj))
        else:
            input_ids_i = self.tokenizer(xi, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            input_ids_j = self.tokenizer(xj, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            embeddings_i = self.model_i(**input_ids_i)
            embeddings_j = self.model_j(**input_ids_j)
        concatenated = torch.cat((embeddings_i, embeddings_j), dim=1)
        return self.dense(concatenated)


class SingleEncoder(torch.nn.Module):
    def __init__(self,
                max_text_length,
                number_of_classification_layers,
                sbert_or_luar):
        super(SingleEncoder, self).__init__()
        self.max_text_length = max_text_length

        if sbert_or_luar == "sbert":
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
            self.tokenizer = None
            hidden_size = self.model[1].word_embedding_dimension
        elif sbert_or_luar == "luar":
            self.model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", padding=True, truncation=True)
            hidden_size = self.model.config.embedding_size
        else:
            raise ValueError("Unsupoorrted model type", sbert_or_luar)
        
        if number_of_classification_layers == 1:
            self.dense = torch.nn.Linear(hidden_size,7)
        else: #IT HAS TO BE 2
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(hidden_size,hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size//2,7)
            )
        self.sbert_or_luar = sbert_or_luar
        self.dense = self.dense.to(self.model.device)
    
    def forward(self, x, _=None): # the last one is to be compatible with the dualencoder
        if self.sbert_or_luar == "sbert":
            embeddings =  torch.from_numpy(self.model.encode(x))
        else:
            input_ids = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            embeddings = self.model(**input_ids)
        
        return self.dense(embeddings)