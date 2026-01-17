import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = {
    "[PAD]": 0,
    "[MASK]": 1,
    "transformers": 2,
    "use": 3,
    "self": 4,
    "attention": 5,
    "mars": 6,
    "is": 7,
    "called": 8,
    "the": 9,
    "red": 10,
    "planet": 11
}

inv_vocab = {v: k for k, v in vocab.items()}

sentences = [
    ["transformers", "use", "[MASK]", "attention"],
    ["mars", "is", "called", "the", "[MASK]", "planet"]
]

targets = [
    4,
    10
]

max_len = 6

def encode(sentence):
    ids = [vocab[w] for w in sentence]
    ids += [vocab["[PAD]"]] * (max_len - len(ids))
    return ids

inputs = torch.tensor([encode(s) for s in sentences]).to(device)
targets = torch.tensor(targets).to(device)

model = TransformerEncoder(
    vocab_size=len(vocab),
    d_model=32,
    num_layers=2,
    num_heads=4,
    dim_ff=64
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(inputs)
    mask_positions = (inputs == vocab["[MASK]"]).nonzero(as_tuple=True)
    masked_outputs = outputs[mask_positions[0], mask_positions[1]]
    loss = criterion(masked_outputs, targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    outputs = model(inputs)
    masked_outputs = outputs[mask_positions[0], mask_positions[1]]
    predictions = torch.argmax(masked_outputs, dim=1)
    for p in predictions:
        print(inv_vocab[p.item()])
