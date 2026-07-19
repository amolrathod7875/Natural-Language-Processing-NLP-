import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import html
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'PAD': 0, 'UNK': 1}
        self.idx2word = {0: 'PAD', 1: 'UNK'}
        self.min_freq = min_freq
    def build_vocab(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        for w, c in counts.items():
            if c >= self.min_freq and w not in self.word2idx:
                idx = len(self.idx2word)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]
    def __len__(self):
        return len(self.idx2word)

class RNNSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.vocab.encode(self.texts[idx])
        if len(enc) < self.max_len: enc += [0] * (self.max_len - len(enc))
        else: enc = enc[:self.max_len]
        return torch.tensor(enc), torch.tensor(self.labels[idx])

def train_rnn_sentiment(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            out = model(texts)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                out = model(texts)
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={correct/total:.4f}')
    return model

if __name__ == '__main__':
    texts = ['great movie loved it', 'terrible waste time', 'amazing story', 'bad film avoid']
    labels = [1, 0, 1, 0]
    vocab = Vocabulary()
    vocab.build_vocab(texts)
    dataset = TextDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = RNNSentiment(len(vocab), 50, 64, 2)
    train_rnn_sentiment(model, loader, loader, epochs=3)
    print(f'Vocab: {len(vocab)}')

