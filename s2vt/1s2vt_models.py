#Non beam_search, data size 1970

import torch
import torch.nn as nn
import torch.nn.functional as F
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
#perbaikan: pack_padded_sequence, beam search

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
        """
        Args:
            video_dir (str): Directory where video files are stored.
            annotations (list of tuples): List of (video_file, caption) pairs.
            vocab (Vocabulary): Vocabulary object for converting words to indices.
            transform (callable, optional): Optional transform to be applied to video frames.
            max_caption_len (int, optional): Maximum caption length. Defaults to 20.
        """
        self.video_dir = video_dir
        self.annotations = annotations  # List of tuples [(video_file, caption), ...]
        self.vocab = vocab
        self.transform = transform
        self.max_caption_len = max_caption_len
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Fetches the video and caption pair at the given index.
        Args:
            idx (int): Index of the video-caption pair.
        Returns:
            Tuple (frames, caption_tensor): Video frames as a tensor and tokenized caption tensor.
        """
        video_file, caption = self.annotations[idx]  # Unpack the tuple (video_file, caption)
        video_path = os.path.join(self.video_dir, video_file + '.avi')

        # Extract frames from video
        frames = self.extract_frames(video_path)
        
        # Preprocess caption: tokenize and numericalize
        caption_tokens = self.tokenize_caption(caption)
        caption_indices = self.vocab.numericalize(caption_tokens)
        caption_indices = [self.vocab.word2idx["<start>"]] + caption_indices + [self.vocab.word2idx["<end>"]]
        caption_indices = self.pad_caption(caption_indices)
        
        # Apply transformations to frames if specified
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        caption_tensor = torch.tensor(caption_indices)

        return torch.stack(frames), caption_tensor

    def tokenize_caption(self, caption):
        """
        Tokenizes a caption by lowercasing and removing punctuation.
        Args:
            caption (str): The caption to tokenize.
        Returns:
            List[str]: List of tokens.
        """
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        return caption.split()

    def pad_caption(self, caption_indices):
        """
        Pads or truncates the caption to the specified maximum length.
        Args:
            caption_indices (List[int]): List of word indices.
        Returns:
            List[int]: Padded or truncated list of word indices.
        """
        if len(caption_indices) >= self.max_caption_len:
            return caption_indices[:self.max_caption_len]
        else:
            return caption_indices + [self.vocab.word2idx["<pad>"]] * (self.max_caption_len - len(caption_indices))

    def extract_frames(self, video_path, target_frames=30):
        """
        Extract frames from a video at regular intervals.
        Args:
            video_path (str): Path to the video file.
            target_frames (int): Number of frames to extract.
        Returns:
            List[torch.Tensor]: List of extracted video frames.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // target_frames)  # Interval for frame extraction
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or extracted_count >= target_frames:
                break
            
            # Only extract frames at the specified interval
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize
                frames.append(frame)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        
        # Ensure we always return exactly `target_frames` frames, padding with the last frame if necessary
        while len(frames) < target_frames:
            frames.append(frames[-1])  # Duplicate the last frame if video has fewer frames

        return frames


def preprocess_annotations(annotation_file):
    annotations = []  # List to hold (video_file, caption) pairs
    sentences = set()  # Set to hold unique captions
    
    with open(annotation_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Only process non-empty lines
                # Find the position of the first space separating video_file and caption
                split_index = line.find(' ')
                if split_index != -1:  # Ensure there's a space for separation
                    video_file = line[:split_index]
                    caption = line[split_index + 1:]

                    # Append a tuple for each video file and caption
                    annotations.append((video_file, caption))
                    sentences.add(caption)  # Add caption to the set for uniqueness
                else:
                    print(f"Warning: Line does not contain a space separator: {line}")
            else:
                print("Warning: Empty line encountered.")
    
    return annotations, list(sentences)  # Convert set to list for the return value

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}")
    return epoch, loss


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=4,batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        #print("size of input EncoderRNN: ",input.size())
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Pre-trained CNN for Video Frame Feature Extraction
class PretrainedCNN(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, output_size=512):
        super(PretrainedCNN, self).__init__()

        # Select the pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove classifier
            in_features = self.model.fc.in_features
        
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.model.features.children()))  # Use only feature extractor
            in_features = 512 * 7 * 7  # Adjust output size of VGG16 according to input size

        elif model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.fc.in_features

        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.classifier[1].in_features

        # A fully connected layer to map CNN output to desired output size
        self.fc = nn.Linear(in_features, output_size)
    
    def forward(self, x):
        # Extract features from the CNN (trainable weights)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Linear layer to get the desired output size
        return x


# Luong Attention class
class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, keys):
        query = self.attn(query)  # General attention
        scores = torch.bmm(query, keys.permute(0, 2, 1))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, keys)
        return context, attn_weights
    

class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,max_len=20):
        super(DecoderAttention, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = LuongAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size,num_layers=4, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # print("encoder_outputs.size: ",encoder_outputs.size())
        # print("encoder_hidden.size: ",encoder_hidden.size())
        #print("target_tensor size: ",target_tensor.size())
        #decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_("<start>")
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(vocab.word2idx["<start>"])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        for i in range(self.max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, hidden_size]
        query = hidden[-1].unsqueeze(1)  # Only use the last layer's hidden state for attention, and keep batch dimension
        context, attn_weights = self.attention(query, encoder_outputs)  # [batch_size, 1, hidden_size]
        #embedded = embedded.repeat(1, self.gru.num_layers, 1)  # [batch_size, num_layers, hidden_size
        # print("embedded.size: ",embedded.size())
        # print("context.size: ",context.size())
        input_gru = torch.cat((embedded, context), dim=2)  # [batch_size, num_layers, 2 * hidden_size]
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

    # def forward_step(self, input, hidden, encoder_outputs):
    #     embedded = self.dropout(self.embedding(input))

    #     query = hidden.permute(1, 0, 2)
    #     context, attn_weights = self.attention(query, encoder_outputs)
    #     #embedded = embedded.repeat(1, self.gru.num_layers, 1) 
    #     print("embedded.size: ",embedded.size())
    #     print("context.size: ",context.size())
    #     input_gru = torch.cat((embedded, context), dim=2)

    #     output, hidden = self.gru(input_gru, hidden)
    #     output = self.out(output)

    #     return output, hidden, attn_weights

class VideoAnalysisModel(nn.Module):
    def __init__(self, cnn_model_name, cnn_output_size, hidden_size, output_size):
        super(VideoAnalysisModel, self).__init__()
        self.cnn = PretrainedCNN(model_name=cnn_model_name, output_size=cnn_output_size)
        self.encoder = EncoderRNN(cnn_output_size, hidden_size)
        self.decoder = DecoderAttention(hidden_size, output_size)

    def forward(self, video_frames, target_tensor=None):
        batch_size, num_frames, channels, height, width = video_frames.size()

        cnn_features = []
        for i in range(num_frames):
            frame_features = self.cnn(video_frames[:, i, :, :, :])  # Apply CNN on each frame
            cnn_features.append(frame_features.unsqueeze(1))  # Add frame features with sequence dimension

        cnn_features = torch.cat(cnn_features, dim=1)  # (batch_size, seq_len, feature_size)

        encoder_output, encoder_hidden = self.encoder(cnn_features)
        decoder_output, decoder_hidden, attn_weights = self.decoder(encoder_output, encoder_hidden, target_tensor)

        return decoder_output, attn_weights

def train_model(model, dataloader, criterion, optimizer, num_epochs, clip = 5,checkpoint_path=None):
    start_epoch = 0
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     # Load checkpoint if it exists
    #     start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    loss_arr = []
    model.train()
    print("start from: ",start_epoch)
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for video_frames, captions in tqdm(dataloader):
            video_frames = video_frames.to(device)  # Move video frames to GPU (if available)
            captions = captions.to(device)  # Move captions to GPU (if available)
            
            # start_tokens = torch.full((captions.size(0), 1), vocab.word2idx["<start>"], dtype=torch.long, device=device)  # Tensor untuk <start>
            # end_tokens = torch.full((captions.size(0), 1), vocab.word2idx["<end>"], dtype=torch.long, device=device)  # Tensor untuk <end>
            # input_captions = torch.cat((start_tokens, captions[:, :-1]), dim=1)  # Tambahkan <start> dan ambil semua kecuali token terakhir
            # # Target captions: tambahkan <end> token di akhir
            # target_captions = torch.cat((captions[:, 1:], end_tokens), dim=1)
            input_captions = captions[:,:]
            target_captions = captions[:,:]
            # print("input_caption size train: ",input_captions.size())
            # print("target_caption size train: ",target_captions.size())
            # Forward pass
            outputs, attn_weights = model(video_frames, input_captions)
            
            # Compute loss, reshape outputs and target captions to have the same shape
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * max_caption_len, vocab_size)
            target_captions = target_captions.contiguous().view(-1)  # (batch_size * max_caption_len)
            
            loss = criterion(outputs, target_captions)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Optimizer step
            optimizer.step()
            
            total_loss += loss.item()
            
        # Save checkpoint after every epoch
        # checkpoint_path = f'/home/arifadh/Desktop/checkpoint/s2vt_plain/checkpoint_epoch_{epoch+1}.pth'
        # save_checkpoint(model, optimizer, epoch+1, total_loss/len(dataloader), checkpoint_path)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')
        loss_arr.append(total_loss/len(dataloader))
        print(loss_arr)


def evaluate_video_to_text(model, video_frames, vocab, max_len=30):
    with torch.no_grad():
        # Move video frames to device (GPU/CPU)
        video_frames = video_frames.to(device)

        # Pass the video frames through the model
        decoder_outputs, attn_weights = model(video_frames)

        # Select the most likely words for each time step
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze(-1)

        # Convert token ids to words
        decoded_words = []
        for idx in decoded_ids[0]:  # Assuming batch size = 1 during evaluation
            if idx.item() == vocab.word2idx['<end>']:
                decoded_words.append('<end>')
                break
            decoded_words.append(vocab.idx2word[idx.item()])

    return decoded_words, attn_weights

##pad packed sequence sama beam searching
            
video_dir = '/home/arifadh/Desktop/Dataset/YouTubeClips'
annotation_file = '/home/arifadh/Desktop/Dataset/clean_caption.txt'
#checkpoint_path = '/home/arifadh/Desktop/checkpoint/s2vt_plain/checkpoint_epoch_50.pth'
    # Preprocess annotations
annotations, sentences = preprocess_annotations(annotation_file)
#print("annot: ", annotations) #annotations={video_name:captions}
#print("sentence: ",sentences)
    # Build vocabulary
vocab = Vocabulary(freq_threshold=1)
vocab.build_vocabulary(sentences)
    # Create dataset and dataloader
    #self, video_dir, annotations, vocab, transform=None, max_frames=30, target_frame_size=(240, 320), target_caption_length=10
#dataset = VideoDataset(video_dir, annotations, vocab, max_frames = 50, target_frame_size=(224, 224),target_caption_length=30)
#(self, video_dir, annotations, vocab, transform=None, max_caption_len=20)
dataset = VideoDataset(video_dir,annotations,vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1) #batch ganti jadi 1
evaldata = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1) 
embed_size = 512
hidden_size = 512
vocab_size = len(vocab)
# encoder = Encoder(embed_size).to(device)
# decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size).to(device)

model = VideoAnalysisModel(cnn_model_name="resnet50", cnn_output_size=512, hidden_size=512, output_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
optimizer = optim.Adam(list(model.parameters()), lr=1e-4)

    # Train model
num_epochs = 20
# print("train until: ",num_epochs)
train_model(model, dataloader, criterion, optimizer, num_epochs, checkpoint_path=None)


# Misalkan video_frames berukuran (batch_size, num_frames, channels, height, width)
#terdapat dataloader diatas, bagaimana cara mengambil sebuah video?
cnt = 0
for video_frames, caption in dataloader:
    # video_frames akan berukuran (1, num_frames, channels, height, width) karena batch_size = 1
    video_frames = video_frames.to(device)  # Pindahkan ke device (GPU/CPU)

    # Decode caption untuk satu video menggunakan model
    decoded_sentence, attn_weights = evaluate_video_to_text(model, video_frames, vocab)
    
    # Konversi tensor caption menjadi teks menggunakan vocab
    ground_truth = []
    for idx in caption[0]:  # caption[0] karena batch_size = 1
        word = vocab.idx2word[idx.item()]
        if word == '<end>':  # Stop jika menemukan token <end>
            break
        ground_truth.append(word)
    
    # Print ground truth dan generated caption
    print("Ground Truth: ", " ".join(ground_truth))
    print("Generated Caption: ", " ".join(decoded_sentence))
    cnt=cnt+1
    if cnt == 50:
        break  # Hanya mengambil satu batch (satu video) # Hanya mengambil satu video, jadi kita break loop
