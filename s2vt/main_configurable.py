import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
import string
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Special tokens
        self.add_word("<pad>")
        self.add_word("<start>")
        self.add_word("<end>")
        self.add_word("<unk>")
    
    def __len__(self):
        return len(self.word2idx)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def numericalize(self, text):
        # Convert a list of words into their corresponding indices
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text]
    
    def denumericalize(self, indices):
        # Convert indices back to words
        return [self.idx2word[idx] for idx in indices]

    def build_vocabulary(self, sentences):
        word_freq = {}
        for sentence in sentences:
            for word in sentence.split():
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        
        for word, freq in word_freq.items():
            if freq >= self.freq_threshold:
                self.add_word(word)

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

# Dataset for video and caption
class VideoDataset(Dataset):
    def __init__(self, video_dir, annotations, vocab, transform=None, max_caption_len=20):
        self.video_dir = video_dir
        self.annotations = annotations
        self.vocab = vocab
        self.transform = transform
        self.video_files = list(annotations.keys())
        self.max_caption_len = max_caption_len
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        caption = self.annotations[video_file]
        video_path = os.path.join(self.video_dir, video_file + '.avi')
        print("Video Path is: ",video_path)
        print("Captions: ",caption)
        frames = self.extract_frames(video_path)
        
        # Preprocess caption: tokenize and numericalize
        caption_tokens = self.tokenize_caption(caption)
        caption_indices = self.vocab.numericalize(caption_tokens)
        caption_indices = [self.vocab.word2idx["<start>"]] + caption_indices + [self.vocab.word2idx["<end>"]]
        caption_indices = self.pad_caption(caption_indices)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        return torch.stack(frames), torch.tensor(caption_indices)

    def tokenize_caption(self, caption):
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        return caption.split()

    def pad_caption(self, caption_indices):
        if len(caption_indices) >= self.max_caption_len:
            return caption_indices[:self.max_caption_len]
        else:
            return caption_indices + [self.vocab.word2idx["<pad>"]] * (self.max_caption_len - len(caption_indices))

    # Function to extract frames from video
    def extract_frames(self, video_path, target_frames=30):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total jumlah frame dalam video
        frame_interval = max(1, total_frames // target_frames)  # Interval untuk mengekstrak frame penting
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or extracted_count >= target_frames:
                break
            
            # Hanya ambil frame jika sesuai interval
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        return frames




class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size,cnn_type = "resnet50", rnn_type = "gru", num_heads = 8, num_layers=1):
        super(Encoder, self).__init__()
        
        # CNN (ResNet) untuk feature extraction dari frame video
        if cnn_type == "vgg16":
            vgg = models.vgg16(pretrained=True)
            self.cnn = nn.Sequential(*list(vgg.features.children()))  # Only convolutional layers
            self.cnn_fc = nn.Linear(512, embed_size)
        elif cnn_type == "resnet50":
            resnet = models.resnet50(pretrained=True)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Menghapus lapisan fully connected
            self.cnn_fc = nn.Linear(resnet.fc.in_features, embed_size)  # Lapisan fully connected untuk project feature
        else:
            raise ValueError(f"Unknown backbone: {cnn_type}")

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
            self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        elif rnn_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
    def forward(self, frames):
        batch_size, num_frames, C, H, W = frames.size()
        
        # CNN feature extraction for each frame
        cnn_features = []
        for i in range(num_frames):
            frame_feat = self.cnn(frames[:, i])  # Extract features from CNN
            frame_feat = frame_feat.mean([2, 3])  # Global Average Pooling
            frame_feat = self.cnn_fc(frame_feat)  # Project to embedding size
            cnn_features.append(frame_feat)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, num_frames, embed_size)
        print("CNN out enc: ",cnn_features.size())
        # Apply RNN or Transformer based on the type chosen
        if hasattr(self, 'rnn'):
            rnn_out, _ = self.rnn(cnn_features)  # (batch_size, num_frames, hidden_size)
            attn_output, _ = self.multihead_attn(rnn_out, rnn_out, rnn_out)  # (batch_size, num_frames, hidden_size)
            print("RNN size enc: ", attn_output.size())
            return attn_output
        
        elif hasattr(self, 'transformer_encoder'):
            transformer_out = self.transformer_encoder(cnn_features)  # (batch_size, num_frames, embed_size)
            print("TNN size enc: ", transformer_out.size())
            return transformer_out  # Mengembalikan hidden state terakhir juga untuk decoding

#Decoder bener bos
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, rnn_type='gru', num_layers=3, num_heads=8, max_seq_length=20):
        super(Decoder, self).__init__()
        
        self.rnn_type = rnn_type
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if rnn_type in ['lstm','gru']:
            self.hidden_size = hidden_size
            #self.encoder_dim = encoder_dim
        # Pilih RNN (LSTM atau GRU)
            if rnn_type == 'lstm':
                self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            elif rnn_type == 'gru':
                self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
            else:
                raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
            # Multi-Head Attention
            self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
            # Final output layer untuk prediksi token
            self.fc = nn.Linear(hidden_size, vocab_size)
        else:
            if rnn_type == 'transformer':
                self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
                decoder_layer = TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
                self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        # Embed captions
        embeddings = self.embed(captions)  # (batch_size, max_seq_length, embed_size)
        #if rnn is lstm/gru
        if self.rnn_type in ['lstm','gru']:
            if self.rnn_type == 'lstm':
                h, c = self.init_hidden(batch_size)
            elif self.rnn_type == 'gru':
                h = self.init_hidden_gru(batch_size)
            
            outputs = []
            # Loop through each time step
            for t in range(captions.size(1)):
                # Step melalui waktu, gunakan LSTM/GRU untuk menghasilkan hidden state
                rnn_input = embeddings[:, t].unsqueeze(1)  # (batch_size, 1, embed_size)
                if self.rnn_type == 'lstm':
                    out, (h, c) = self.rnn(rnn_input, (h, c))  # LSTM
                elif self.rnn_type == 'gru':
                    out, h = self.rnn(rnn_input, h)  # GRU
                
                # Multihead attention
                attn_output, attn_weights = self.multihead_attn(out, encoder_out, encoder_out)
                
                # Predict vocabulary
                output = self.fc(attn_output.squeeze(1))  # (batch_size, vocab_size)
                outputs.append(output)
            
            # Stack outputs to get final size (batch_size, max_seq_length, vocab_size)
            outputs = torch.stack(outputs, dim=1)  # (batch_size, max_seq_length, vocab_size)
            return outputs
        elif self.rnn_type == 'transformer':
            embeddings += self.positional_encoding[:, :captions.size(1)]
            
            encoder_out = encoder_out.transpose(0, 1)  # (num_frames, batch_size, embed_size)
            embeddings = embeddings.transpose(0, 1)
            
            transformer_out = self.transformer_decoder(embeddings,encoder_out)
            transformer_out = transformer_out.transpose(0, 1)
            output = self.fc(transformer_out)
            return output
            
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states for LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),  # h_0
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))  # c_0

    def init_hidden_gru(self, batch_size):
        # Initialize hidden state for GRU
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)  # h_0
    #add transformer
    def generate_caption(self, encoder_out, vocab, max_seq_length=20):
        # batch_size = encoder_out.size(0)
        
        if self.rnn_type in ['lstm','gru']:
            if self.rnn_type == 'lstm':
                h, c = self.init_hidden(1)
            elif self.rnn_type == 'gru':
                h = self.init_hidden_gru(1)
        
        # Start token input
        inputs = torch.tensor([[vocab['<start>']]]).to(device)
        embeddings = self.embed(inputs)
        caption = []
        
        for t in range(max_seq_length):
            # Step RNN (LSTM/GRU)
            rnn_input = embeddings[:, 0].unsqueeze(1)  # (batch_size, 1, embed_size)
            # print("rnn_input eval: ",rnn_input.size())
            if self.rnn_type == 'lstm':
                out, (h, c) = self.rnn(rnn_input, (h, c))
            elif self.rnn_type == 'gru':
                out,h = self.rnn(rnn_input, h)
            #print("encoder_out eval: ",encoder_out.size())
            # Apply multi-head attention
            single_frame = encoder_out[0:1, t:t+1, :]
            attn_output, attn_weights = self.multihead_attn(out, single_frame, single_frame)
            
            # Predict next word
            output = self.fc(attn_output.squeeze(1))
            predicted = output.argmax(1)
            
            caption.append(predicted.item())
            if predicted.item() == vocab['<end>']:
                break
            
            # Update the input for the next time step with predicted word
            inputs = predicted.unsqueeze(1)
            embeddings = self.embed(inputs)
        
        return caption



def preprocess_annotations(annotation_file):
    annotations = {}
    sentences = []
    with open(annotation_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Hanya proses baris yang tidak kosong
                # Temukan posisi spasi pertama yang memisahkan video_file dan caption
                split_index = line.find(' ')
                if split_index != -1:  # Pastikan ada spasi untuk pemisahan
                    video_file = line[:split_index]
                    caption = line[split_index + 1:]
                    annotations[video_file] = caption
                    sentences.append(caption)
                else:
                    print(f"Warning: Line does not contain a space separator: {line}")
            else:
                print("Warning: Empty line encountered.")
    return annotations, sentences

def save_checkpoint(encoder, decoder, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

# Function to load checkpoint
def load_checkpoint(encoder, decoder, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}")
    return epoch, loss


def train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs, vocab_size, checkpoint_path=None):
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load checkpoint if it exists
        start_epoch, _ = load_checkpoint(encoder, decoder, optimizer, checkpoint_path)
    loss_arr = []
    encoder.train()
    decoder.train()
    print("start from: ",start_epoch)
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for frames, captions in tqdm(dataloader):
            frames, captions = frames.to(device), captions.to(device)
            #print("type of captions: ",type(captions))
            #print(captions)
            optimizer.zero_grad()
            # print("frames size: ",frames.size())   #batch size caption sama frame beda?
            # print("captions size: ",captions.size())
            # Forward pass
            encoder_out = encoder(frames)
            #print("size of encoder_out", encoder_out.size())
            outputs = decoder(encoder_out, captions[:, :-1])  # input captions without <end> token
            # print("decoder output size:", outputs.size())
            # print("captions size:", captions[:,1:].size())
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))  # compare with captions without <start> token
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Save checkpoint after every epoch
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        save_checkpoint(encoder, decoder, optimizer, epoch+1, total_loss/len(dataloader), checkpoint_path)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')
        loss_arr.append(total_loss/len(dataloader))
        print(loss_arr)


#reference ama captions di eval kok beda ya?
# def evaluate_model(encoder, decoder, dataloader, vocab, checkpoint_path=None):
    
#     bleu_scores = []
#     # if checkpoint_path and os.path.exists(checkpoint_path):
#     #     # Load checkpoint if it exists
#     #     start_epoch, _ = load_checkpoint(encoder, decoder, optimizer, checkpoint_path)

#     encoder.eval()
#     decoder.eval()
    
#     with torch.no_grad():
#         for frames, captions in tqdm(dataloader):
#             frames = frames.to(device)
#             encoder_out = encoder(frames)
#             generated_caption = decoder.generate_caption(encoder_out, vocab)
#             #reference = [captions[0].split()]  # ground truth, yang ini masih salah
#             # Misalkan 'captions' adalah tensor dengan bentuk (1, max_seq_length)
#             print("captions size: ",captions.size())
#             caption_tensor = captions[0]  # Ambil caption pertama
#             reference = [' '.join([vocab.idx2word[idx.item()] for idx in caption_tensor if idx.item() != 0])]  # Menghindari 0 (padding)
#             print("reference: ",reference)
#             candidate = [vocab.idx2word[word] for word in generated_caption]
#             print("candidate: ",candidate)
#             bleu = sentence_bleu(reference, candidate)
#             print("BLEU: ",bleu)
#             bleu_scores.append(bleu)
    
#     avg_bleu = sum(bleu_scores) / len(bleu_scores)
#     print(f'Average BLEU score: {avg_bleu}')

def evaluate_model(encoder, decoder, dataloader, vocab, checkpoint_path=None):
    bleu_scores = []
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for frames, captions in tqdm(dataloader):
            frames = frames.to(device)
            captions = captions.to(device)  # Pastikan caption juga ada di perangkat yang sama
            encoder_out = encoder(frames)
            generated_caption = decoder.generate_caption(encoder_out, vocab)

            # Extract the ground truth reference caption
            caption_tensor = captions[0]  # Ambil caption pertama dari batch
            reference = [vocab.denumericalize(caption_tensor.tolist())]  # Konversi indeks caption menjadi kata
            reference = ' '.join(reference[0]).replace("<start>", "").replace("<end>", "").strip()  # Hilangkan token start/end
            print("reference: ", reference)

            candidate = [vocab.idx2word[word] for word in generated_caption]  # Caption hasil prediksi
            print("candidate: ", candidate)

            # Hitung skor BLEU
            bleu = sentence_bleu([reference.split()], candidate)
            print("BLEU: ", bleu)
            bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f'Average BLEU score: {avg_bleu}')


video_dir = '/home/arifadh/Desktop/Dataset/YouTubeClips'
annotation_file = '/home/arifadh/Desktop/Dataset/clean_caption.txt'

#checkpoint_path = '/home/arifadh/Desktop/checkpoint/checkpoint_epoch_35.pth'
checkpoint_path = '/home/arifadh/Desktop/Skripsi-Magang/checkpoint_epoch_1.pth'
    # Preprocess annotations
annotations, sentences = preprocess_annotations(annotation_file)
#print("annot: ", annotations) #annotations={video_name:captions}
#print("sentence: ",sentences)
    # Build vocabulary
vocab = Vocabulary(freq_threshold=2)
vocab.build_vocabulary(sentences)

    # Create dataset and dataloader
dataset = VideoDataset(video_dir, annotations, vocab, max_caption_len=30)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1) #batch ganti jadi 1
eval_data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize encoder, decoder, criterion, and optimizer
embed_size = 512
hidden_size = 512
vocab_size = len(vocab)
# encoder = Encoder(embed_size).to(device)
# decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size).to(device)
encoder = Encoder(embed_size, hidden_size).to(device)
decoder = Decoder(embed_size, hidden_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    # Train model
num_epochs = 1
train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs, vocab_size,checkpoint_path = checkpoint_path)

    # Evaluate model
evaluate_model(encoder, decoder, eval_data, vocab)
