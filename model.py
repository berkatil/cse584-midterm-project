from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualEncoder(torch.nn.Module):
    def __init__(self, max_xi_length, max_xj_length, number_of_classification_layers, sbert_or_luar):
        super(DualEncoder, self).__init__()
        self.max_xi_length = max_xi_length
        self.max_xj_length = max_xj_length

        if sbert_or_luar == "sbert":
            self.model_i = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
            self.model_j = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
            hidden_size = self.model_i.get_sentence_embedding_dimension()
            self.tokenizer = None
        elif sbert_or_luar == "luar":
            self.model_i = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True).to(device)
            self.model_j = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True).to(device)
            hidden_size = self.model_i.config.hidden_size
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")
        else:
            raise ValueError("Unsupported model type", sbert_or_luar)

        if number_of_classification_layers == 1:
            self.dense = torch.nn.Linear(hidden_size * 2, 7)
        else:
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 7),
            )
        self.dense = self.dense.to(device)
        self.sbert_or_luar = sbert_or_luar

    def forward(self, xi, xj):
        if self.sbert_or_luar == "sbert":
            embeddings_i = torch.tensor(self.model_i.encode(xi)).to(device)
            embeddings_j = torch.tensor(self.model_j.encode(xj)).to(device)
        else:
            input_ids_i = self.tokenizer(xi, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids_j = self.tokenizer(xj, return_tensors="pt", padding=True, truncation=True).to(device)
            embeddings_i = self.model_i(**input_ids_i).pooler_output.to(device)
            embeddings_j = self.model_j(**input_ids_j).pooler_output.to(device)
        
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
            hidden_size = self.model.get_sentence_embedding_dimension()
        elif sbert_or_luar == "luar":
            self.model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", padding=True, truncation=True)
            hidden_size = self.model.config.hidden_size
        else:
            raise ValueError("Unsupported model type", sbert_or_luar)
        
        if number_of_classification_layers == 1:
            self.dense = torch.nn.Linear(hidden_size, 7)
        else:  
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size // 2, 7)
            )
        self.sbert_or_luar = sbert_or_luar
        self.dense = self.dense.to(device)
    
    def forward(self, x, _=None):  # the last one is to be compatible with the dualencoder
        if self.sbert_or_luar == "sbert":
            embeddings = torch.from_numpy(self.model.encode(x)).to(device)
        else:
            input_ids = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(device)
            embeddings = self.model(**input_ids).last_hidden_state.to(device)
        
        return self.dense(embeddings)