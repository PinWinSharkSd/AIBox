"""
    AIBox Tech.
    is a very simple AI Text Generator Engine.
    By Ali Hosain Ali
    Tell me your suggestion on (alihosainale@gmail.com)
"""
# v0.06 Pure PyTorch FNet Version
# 2025/05/04

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from difflib import get_close_matches
import sys
import pickle
from torch.utils.data import Dataset, DataLoader

# Pure PyTorch implementation of the FNet block
class FNetBlock(nn.Module):
    def __init__(self, dim, d_ff=None):
        """
        A single FNet block implemented using pure PyTorch.
        Based on the paper "FNet: Mixing Tokens with Fourier Transforms".
        Args:
            dim: The input and output dimension of the block (d_model).
            d_ff: The dimension of the feed-forward network. If None, defaults to 4 * dim.
        """
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff if d_ff is not None else 4 * dim

        # Layer Normalization before the Fourier mixing
        self.norm1 = nn.LayerNorm(dim)

        # No parameters for Fourier Transform, just the operation
        # We'll use torch.fft.fft2 for 2D Fourier Transform across sequence and feature dimensions

        # Layer Normalization after the Fourier mixing and before the feed-forward
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, self.d_ff),
            nn.GELU(), # Using GELU activation as commonly used in modern transformers
            nn.Linear(self.d_ff, dim)
        )

    def forward(self, x):
        """
        Forward pass for the FNet block.
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        # Residual connection 1
        residual = x

        # Apply LayerNorm
        x = self.norm1(x)

        # Apply 2D Fast Fourier Transform
        # FFT across the sequence length dimension (dim=1) and the feature dimension (dim=2)
        # torch.fft.fft2 computes 2D FFT. It operates on the last two dimensions by default.
        # If input is (batch, seq_len, dim), fft2 will operate on (seq_len, dim).
        x = torch.fft.fft2(x, dim=(-2, -1)).real # Take the real part as in the original FNet paper

        # Add residual connection 1
        x = x + residual

        # Residual connection 2
        residual = x

        # Apply LayerNorm
        x = self.norm2(x)

        # Apply Feed-forward network
        x = self.ff(x)

        # Add residual connection 2
        x = x + residual

        return x

# Pure PyTorch FNet Language Model
class FNet_Model_PureTorch(nn.Module):
    def __init__(self, vocab_size=50, embed_size=50, d_model=50, d_ff=200, n_layers=2, seq_len=5):
        """
        FNet Model for text generation implemented using pure PyTorch.
        Args:
            vocab_size: Size of the vocabulary.
            embed_size: Dimension of the word embeddings.
            d_model: The dimension of the model's internal representation.
            d_ff: The dimension of the feed-forward network in FNet layers.
            n_layers: Number of FNet layers.
            seq_len: The expected input sequence length.
        """
        super(FNet_Model_PureTorch, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        # Project embedding dimension to d_model if they are different
        self.embedding_projection = nn.Linear(embed_size, d_model) if embed_size != d_model else nn.Identity()

        # Stack FNet blocks
        self.fnet_layers = nn.Sequential(
            *[FNetBlock(dim=d_model, d_ff=d_ff) for _ in range(n_layers)]
        )

        # Output layer maps from d_model (last token's representation) to vocab_size
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.25)
        self.seq_len = seq_len # Store seq_len for generation
        self.d_model = d_model # Store d_model

    def forward(self, x):
        """
        Forward pass for the FNet model.
        Args:
            x: Input tensor of shape (batch, seq_len) containing token IDs.
        Returns:
            Output tensor of shape (batch, vocab_size) for the predicted next token.
        """
        # x: (batch, seq_len)
        x = self.embed(x)  # (batch, seq_len, embed_size)

        # Project embedding if embed_size != d_model
        x = self.embedding_projection(x) # (batch, seq_len, d_model)

        x = self.dropout(x) # Apply dropout after embedding/projection

        # Pass through FNet layers
        # FNet layers maintain the shape (batch, seq_len, d_model)
        x = self.fnet_layers(x)

        # We take the output corresponding to the last token in the sequence
        # to predict the next token.
        out = self.out(x[:, -1, :]) # (batch, d_model) -> (batch, vocab_size)

        return out

class SubwordTokenizer:
    def __init__(self, vocab):
        """
        vocab: قائمة المفردات المستخدمة في النموذج.
        """
        # التأكد من وجود رمز للمجهول "<unk>"
        if "<unk>" not in vocab:
            vocab.append("<unk>")
        vocab = set(vocab)
        self.vocab = vocab
        # Start indexing from 0 for compatibility with nn.Embedding
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        print("Embidings:", len(vocab))

    def encode(self, text):
        """
        تحويل النص إلى قائمة من الأرقام باستخدام القاموس.
        """
        encoded = []
        tokens = text.split()
        # Use 0 for <unk> if it exists, otherwise handle missing words
        unk_id = self.word2idx.get("<unk>", None)
        for token in tokens:
            if token in self.word2idx:
                encoded.append(self.word2idx[token])
            else:
                # استخدام get_close_matches مع التحقق من النتيجة
                # Note: get_close_matches works on keys, which are words
                matches = get_close_matches(token, self.word2idx.keys(), n=1, cutoff=0.6) # Increased cutoff slightly
                if matches:
                    encoded.append(self.word2idx[matches[0]])
                elif unk_id is not None:
                     # استخدام رمز المجهول إذا لم يوجد أي تطابق
                    encoded.append(unk_id)
                else:
                    # Fallback if <unk> is not in vocab and no match found
                    print(f"Warning: Token '{token}' not found in vocabulary and no close match. Using index 0.")
                    encoded.append(0) # Default to 0 if <unk> not available

        return encoded

    def decode(self, token_ids):
        """
        تحويل قائمة من الأرقام إلى نص باستخدام القاموس.
        """
        # Use get with a default value (e.g., "<unk>") in case of unknown IDs
        words = [self.idx2word.get(idx, "<unk>") for idx in token_ids]
        return ' '.join(words)

# تعريف Dataset مخصص للنصوص مع استخدام تسلسلات أطول لتعلم السياق
class TextDataset(Dataset):
    def __init__(self, dt, seq_len=5):
        """
        dt: قائمة من الأرقام (النص المشفر)
        seq_len: طول التسلسل المستخدم للتدريب
        """
        self.dt = dt
        self.seq_len = seq_len

    def __len__(self):
        # Ensure there are enough tokens to form at least one sequence
        return max(0, len(self.dt) - self.seq_len)

    def __getitem__(self, idx):
        # x is the sequence of tokens up to the last one
        x = self.dt[idx: idx + self.seq_len]
        # y is the target token, which is the one immediately following the sequence x
        y = self.dt[idx + self.seq_len] # Predict the token *after* the sequence

        # Ensure x has the correct length, pad if necessary (though TextDataset structure should prevent this)
        # If padding was needed, a padding token ID (e.g., 0) would be used.
        # For this dataset structure, x will always be seq_len long.

        return torch.tensor(x, dtype=torch.long), torch.tensor([y], dtype=torch.long) # Target y is a single token ID

# الكلاس الذي يحاكي واجهة aibox الموجودة في كود chainer
class aibox:
    def __init__(self):
        self.model = None
        self.reload_data = [] # Seems unused in this version
        self.training_losses = []  # لتخزين خسائر كل إبوك
        self.vc = [] # Vocabulary list
        self.cvc = [] # Encoded text (corpus)
        self.tk = None # Tokenizer instance
        self.vocab_size = 0 # Size of the vocabulary
        self.seq_len = 5 # Default sequence length
        self.current_loss = 0.0 # To display in progress bar
        # Store model hyperparameters to recreate the model correctly when loading
        self.model_params = {
            'embed_size': 100,
            'd_model': 100,
            'd_ff': 400,
            'n_layers': 2,
            'seq_len': 5
        }


    def progress_bar(self, current, total, ti, bar_length=22):
        """عرض شريط التقدم الأخضر أثناء التدريب"""
        percent = current / total
        filled_length = int(bar_length * percent)
        green = "\033[92m"
        reset = "\033[0m"
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        loss_str = f"{self.current_loss:.5f}"
        sys.stdout.write(f"\r{green}|{bar}| {int(percent * 100)}%{reset} (Loss: {loss_str}) {ti}")
        sys.stdout.flush()

    def init_data(self, d):
        # تقسيم النص إلى كلمات وبناء قاموس جديد مع تضمين <unk>
        new_vocab = d.split()
        # Merge new vocab with existing, preserving order and uniqueness
        merged_vocab = list(dict.fromkeys(self.vc + new_vocab))
        self.vc = merged_vocab
        # Vocab size includes the <unk> token if it's added by the tokenizer
        # The tokenizer handles adding <unk> and setting up word2idx/idx2word
        self.tk = SubwordTokenizer(self.vc)
        self.vocab_size = len(self.tk.vocab) # Get actual vocab size from tokenizer
        self.cvc = self.tk.encode(d)
        print(f"Initialized data. Vocabulary size: {self.vocab_size}, Corpus length: {len(self.cvc)}")

    def custom_model(self, vocab_size=None, embed_size=100, d_model=100, d_ff=400, n_layers=2, seq_len=5):
        """
        Create or replace the model with a Pure PyTorch FNet model.
        Args are stored and passed to the FNet_Model_PureTorch constructor.
        """
        if vocab_size is None:
            # Use the vocabulary size determined during init_data
            vocab_size = self.vocab_size
            if vocab_size == 0:
                 print("Warning: Vocabulary size is 0. Please call init_data first.")
                 # Default to a small size if data not initialized, though training will fail
                 vocab_size = 50

        # Store the model parameters
        self.model_params = {
            'vocab_size': vocab_size,
            'embed_size': embed_size,
            'd_model': d_model,
            'd_ff': d_ff,
            'n_layers': n_layers,
            'seq_len': seq_len
        }
        self.seq_len = seq_len # Also store seq_len directly for convenience

        # Instantiate the FNet_Model_PureTorch
        self.model = FNet_Model_PureTorch(
            vocab_size=self.model_params['vocab_size'],
            embed_size=self.model_params['embed_size'],
            d_model=self.model_params['d_model'],
            d_ff=self.model_params['d_ff'],
            n_layers=self.model_params['n_layers'],
            seq_len=self.model_params['seq_len']
        )
        print(f"Created Pure PyTorch FNet model with params: {self.model_params}")


    def fit(self, epochs=1000, lr=0.001, data=None, target=None, model_name="model_file.pt", batch_size=32, seq_len=None, device=None):
        """
        التدريب باستخدام DataLoader لتقسيم البيانات إلى دفعات.
        يتم تغيير ترتيب البيانات في كل إبوك (shuffle=True) للتأكد من تعلم النموذج لكل أجزاء النص.
        """
        if data is not None:
            self.init_data(data)

        # Use seq_len from custom_model if not provided here
        if seq_len is not None:
            self.seq_len = seq_len
            self.model_params['seq_len'] = seq_len # Update stored params
        elif self.seq_len is None or self.seq_len == 0:
             print("Error: seq_len is not set. Please provide it in custom_model or fit.")
             return None # Or raise an error

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Prefer cpu over vulkan for general use

        # Ensure model is created before training
        if self.model is None:
             print("Model not created. Creating default Pure PyTorch FNet model.")
             # Use stored model_params, update vocab_size if init_data was called
             self.model_params['vocab_size'] = self.vocab_size if self.vocab_size > 0 else self.model_params['vocab_size']
             self.custom_model(**self.model_params) # Use stored params

        self.model.to(device) # Move model to device

        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        # Ensure dt is numpy array of correct type
        dt = np.array(self.cvc, dtype=np.int64)

        # Check if corpus is large enough for the given seq_len
        if len(dt) <= self.seq_len:
             print(f"Error: Corpus length ({len(dt)}) is not greater than seq_len ({self.seq_len}). Cannot create dataset.")
             return None # Or raise an error

        dataset = TextDataset(dt, seq_len=self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print('Starting Fit.')
        self.progress_bar(0, epochs, "[Waiting...]")
        for epoch in range(epochs):
            epoch_loss = 0
            start = time.time()
            # No need for mw counter unless debugging
            # mw = 0

            # DataLoader with shuffle=True shuffles data each epoch
            for x_batch, t_batch in dataloader:
                # mw += 1
                # if mw % 10 == 0: print(f"Batch {mw}") # Optional: print batch progress

                x_batch = x_batch.to(device)
                t_batch = t_batch.to(device).squeeze(-1) # Target is a single token ID, remove extra dimension

                optimizer.zero_grad()

                # FNet forward pass does not use or return hidden state
                y = self.model(x_batch) # y shape: (batch_size, vocab_size)

                # Calculate loss. t_batch is (batch_size), y is (batch_size, vocab_size)
                loss = nn.functional.cross_entropy(y, t_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Calculate average loss for the epoch
            self.current_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            self.training_losses.append(self.current_loss)

            time_elapsed = time.time() - start
            minutes, seconds = divmod(int(time_elapsed), 60)
            form = f"[{minutes:02}:{seconds:02}]"
            self.progress_bar(epoch + 1, epochs, form)

            if target is not None and self.current_loss < target:
                print(f"\nTarget loss ({target}) reached. Stopping training.")
                break

        # Ensure progress bar is finalized
        self.progress_bar(epochs, epochs, form)
        print("\nTraining completed.")

        self.save_model(model_name)

        return self.current_loss

    def generate_text(self, start_text, end=[], max_words=10, device=None):
        """توليد كلمات بناءً على الكلمات السابقة باستخدام نموذج FNet."""
        if self.model is None:
            print("Error: Model not loaded or trained.")
            return start_text

        if self.tk is None:
             print("Error: Tokenizer not initialized. Please call init_data first.")
             return start_text

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device) # Ensure model is on the correct device
        self.model.eval() # Set model to evaluation mode

        # Encode the starting text
        text_ids = self.tk.encode(start_text)
        generated_ids = list(text_ids) # Use a list to easily append new tokens
        generated_words = start_text.split()

        # Get the sequence length the model was trained with
        seq_len = self.seq_len if hasattr(self, 'seq_len') and self.seq_len > 0 else 5 # Default if not set

        #print(f"\nGenerating text (max {max_words} words, using seq_len={seq_len})...")

        with torch.no_grad(): # Disable gradient calculation for inference
            for _ in range(max_words):
                # Get the input sequence for the model: the last seq_len tokens
                # Pad if the current sequence is shorter than seq_len
                current_sequence_ids = generated_ids[-seq_len:]
                padding_needed = seq_len - len(current_sequence_ids)

                # Assuming token ID 0 is suitable for padding (<unk> is typically 0).
                # A dedicated padding token and ID is better practice, but for now, use 0.
                padded_sequence_ids = [self.tk.word2idx.get("<unk>", 0)] * padding_needed + current_sequence_ids

                x = torch.tensor([padded_sequence_ids], dtype=torch.long, device=device)

                # FNet forward pass
                y = self.model(x) # y shape: (1, vocab_size)

                # Get the predicted token ID
                predicted_idx = torch.argmax(y, dim=-1).item()

                # Append the predicted token ID to the generated sequence
                generated_ids.append(predicted_idx)

                # Decode the predicted token ID to a word
                new_word = self.tk.idx2word.get(predicted_idx, "<unk>")
                generated_words.append(new_word)

                # Check if the generated word is an end word
                if end and new_word in end:
                    break

        self.model.train() # Set model back to training mode
        return " ".join(generated_words)

    def load_model(self, model_name="model_file.pt", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.load_state_dict(torch.load(model_name, map_location=device))
            self.load_state()
            print("Model Loaded.")
        else:
            print("Please Create a Custom Model")

    def save_state(self, state_name='aibox_state.st'):
        pickle.dump([self.vc, self.cvc, self.tk], open(state_name, 'wb'))

    def save_model(self, model_name='model_file.pt'):
        torch.save(self.model.state_dict(), model_name)
        self.save_state()

    def load_state(self, state_name='aibox_state.st'):
        a = pickle.load(open(state_name, 'rb'))
        self.vc = a[0]
        self.cvc = a[1]
        self.tk = a[2]