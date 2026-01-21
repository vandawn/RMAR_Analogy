import pandas as pd
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import pickle   
import pdb

# Read the file into a dataframe
df = pd.read_csv('entity2textlong.txt', sep='\t',header=None)

model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Rename the columns of df
df.columns = ['ent_id', 'text']
print(df)
# 对 text 列进行编码，并将结果存储在新列 'text_embedding' 中
df['text_embedding'] = df['text'].apply(lambda x: model.encode(x).tolist())

# Convert the text_embedding column to a 2D array
text_embedding = df['text_embedding'].to_list()
text_embedding = torch.tensor(text_embedding)
print(text_embedding)
print(text_embedding.shape)

# 指定保存路径
pth_file_path = '/data/rwan551/code/SNAG_Analogy/embeddings/MCNet-textual.pth'

# 保存嵌入向量到 .pth 文件
torch.save(text_embedding, pth_file_path)

print(f"Text embeddings saved to {pth_file_path}")



# # Print the dataframe
# print(df.head())

# # Load the pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Move the model to the GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# batch_size = 1000  # Adjust this value according to your GPU memory

# # #查看df['text']是否有空值
# # print(df['text'].isnull().sum())
# # #打印空值的行
# # print(df[df['text'].isnull()])


# # Tokenize the text column
# df['text_tokens_short'] = df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))



# # Pad the tokenized sequences to the same length
# max_length = max(df['text_tokens_short'].apply(len))
# df['text_tokens_short'] = df['text_tokens_short'].apply(lambda x: x + [0] * (max_length - len(x)))



# df['text_tokens_long'] = df['text_long'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# # Pad the tokenized sequences to the same length
# max_length = max(df['text_tokens_long'].apply(len))
# df['text_tokens_long'] = df['text_tokens_long'].apply(lambda x: x + [0] * (max_length - len(x)))

# # Convert the tokenized sequences to tensors and generate the name BERT embeddings
# embeddings_short = []
# num_samples = 1000
# embedding_dim = 768
# target_dim = 300
# for i in range(0, len(df), batch_size):
#     batch = df['text_tokens_short'][i:i+batch_size]
#     input_ids = torch.tensor(batch.tolist()).to(device)
#     with torch.no_grad():
#         outputs = model(input_ids)
#         batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
        
#         batch_embeddings = torch.tensor(batch_embeddings)  # Convert numpy array to tensor
        
#         linear_layer = nn.Linear(embedding_dim, target_dim)
#         embeddings_reduced = linear_layer(batch_embeddings)
#         embeddings_short.extend(embeddings_reduced)

# #df['text_embedding_short'] = embeddings_short

# # Convert the embeddings list to a 2D array
# numpy_embeddings = torch.stack(embeddings_short).numpy()
# # Save the embeddings in a new column
# df['text_embedding_short'] = numpy_embeddings.tolist()

# # Save the embeddings as a pkl file
# pkl_file = 'dbp_norm_name.pkl'
# with open(pkl_file, 'wb') as file:
#     pickle.dump(numpy_embeddings, file)


# # 读取 .pkl 文件
# with open(pkl_file, 'rb') as file:
#     data = pickle.load(file)

# print(data)

# # 打印文件内容
# print("Name embedding shape:")
# print(data.shape)

    
# # Convert the tokenized sequences to tensors and generate the char BERT embeddings
# embeddings_long = []
# num_samples = 1000

# for i in range(0, len(df), batch_size):
#     batch = df['text_tokens_long'][i:i+batch_size]
#     input_ids = torch.tensor(batch.tolist()).to(device)
#     with torch.no_grad():
#         outputs = model(input_ids)
#         batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
        
#         batch_embeddings = torch.tensor(batch_embeddings)  # Convert numpy array to tensor
        
#         # linear_layer = nn.Linear(embedding_dim, target_dim)
#         # embeddings_reduced = linear_layer(batch_embeddings)
#         embeddings_long.extend(batch_embeddings)

# #df['text_embedding_long'] = embeddings_long

# # Convert the embeddings list to a 2D array
# numpy_embeddings = torch.stack(embeddings_long).numpy()
# # Save the embeddings in a new column
# df['text_embedding_long'] = numpy_embeddings.tolist()

# # Save the embeddings as a pkl file
# pkl_file = 'dbp_norm_char.pkl'
# with open(pkl_file, 'wb') as file:
#     pickle.dump(numpy_embeddings, file)


# # 读取 .pkl 文件
# with open(pkl_file, 'rb') as file:
#     data = pickle.load(file)

# print(data)
# # 打印文件内容
# print("Char embedding shape:")
# print(data.shape)
