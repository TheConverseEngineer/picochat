from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools
import time
import tqdm

import native
from src.picochat.model import MiniLLM

def prepare_tokens(vocab_size: int, vocab_path: Path, training_data_path: Path):
    # It's vocab_size - 1 because we need to reserve space for the EOS token
    native.train_tokenizer('datasets/bpe/TinyStories-train.txt', str(vocab_path.resolve()), vocab_size - 1)
    native.tokenize(str(vocab_path.resolve()), "datasets/bpe/TinyStories-train.txt", str(training_data_path.resolve()))


max_mem_usage = 0
def update_max_mem_usage():
    global max_mem_usage
    cmem = torch.mps.driver_allocated_memory()
    max_mem_usage = max(max_mem_usage, cmem)

class TokenDataset(Dataset):
    def __init__(
        self, 
        raw_token_file: Path,
        sequence_length: int,
    ):
        super().__init__()
        self._array = np.memmap(str(raw_token_file.resolve()), dtype=np.uint16, mode='r')
        self._num_rows = self._array.size // (sequence_length + 1)
        self._sequence_length = sequence_length

    def __len__(self):
        return self._num_rows

    def __getitem__(self, index):
        start_index = (self._sequence_length + 1) * index
        next_token_index = start_index + self._sequence_length
        return (
            torch.as_tensor(self._array[start_index:next_token_index]),
            torch.as_tensor(self._array[(start_index+1):(next_token_index+1)])
        )


if __name__ == "__main__":
    ### Specify hyperparameters
    vocab_path = Path("datasets/bpe/vocab.txt")
    training_data_path = Path("datasets/bpe/train_tokens.bin")

    device = torch.device('mps')

    vocab_size = 4096
    embedding_size = 384
    num_transformer_blocks = 4
    hidden_ff_dim = 1024
    num_attention_heads = 16
    sequence_length = 1024
    batch_size = 16

    warm_up_duration = 100
    annealing_duration = 2900
    lr_max = 1e-3
    lr_min = 1e-5
    max_grad_norm = 1

    eval_temperature = 0.4

    writer = SummaryWriter('runs/small_llm_fixlr_mps_flash_attention', flush_secs=5)

    ### Load training data, model, optimizer, and LR scheduler
    # prepare_tokens(vocab_size, vocab_path, training_data_path)
    training_dataset = TokenDataset(training_data_path, sequence_length)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    print(f"Total possible epochs: {len(training_dataset) // batch_size}")

    llm = MiniLLM(
        vocab_size, embedding_size, num_transformer_blocks, hidden_ff_dim, num_attention_heads, max_seq_len=sequence_length
    ).to(device)
    llm.train()
    print(llm)

    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr_max, betas=(0.9, 0.95))
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 1/warm_up_duration, 1, warm_up_duration),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, annealing_duration, lr_min),
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: lr_min)
        ],  
        [warm_up_duration, annealing_duration + warm_up_duration]
    )

    ### Training loop!
    start_time = time.time()
    for epoch, (training_inputs, training_outputs) in enumerate(training_dataloader):
        X, y = training_inputs.to(device, dtype=torch.int), training_outputs.to(device, dtype=torch.long)

        optimizer.zero_grad()
        y_pred = llm(X)
        y_pred, y = y_pred.flatten(end_dim=-2), y.flatten()
        loss = loss_fn(y_pred, y)
        update_max_mem_usage()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        writer.add_scalar('lr', lr_schedule.get_last_lr()[0], epoch)
        lr_schedule.step()

        writer.add_scalar("loss", loss.item(), epoch)
        if epoch == 4:
            start_time = time.time()
        elif epoch >= 5:
            writer.add_scalar("fps", (epoch -4) / (time.time() - start_time), epoch)
        
        update_max_mem_usage()
        writer.add_scalar("MPS watermark", max_mem_usage, epoch)
        if epoch > 3000:
            break
    
    print("Done!")
    writer.flush()
    writer.close()

    print("evaluation: ")
    llm.eval()
    for i in range(5):
        print(f"Story {i}:")
        tokens = [376]
        with torch.no_grad():
            for _ in tqdm.tqdm(range(sequence_length)):
                token_tensor = torch.tensor(tokens, device=device, dtype=torch.int)
                token_tensor.unsqueeze_(axis=0)
                logits = llm(token_tensor)[0, len(tokens) - 1] / eval_temperature
                next_token = torch.distributions.Categorical(logits=logits).sample().item()
                tokens.append(next_token)

                if next_token == vocab_size - 1: break
        
        print(tokens)
        print(native.untokenize(str(vocab_path.resolve()), tokens))





