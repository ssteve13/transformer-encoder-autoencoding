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

def encode_user_sentence(sentence):
    raw_words = sentence.split()
    words = []
    ids = []

    for w in raw_words:
        if w.upper() == "[MASK]":
            words.append("[MASK]")
            ids.append(vocab["[MASK]"])
        else:
            w = w.lower()
            if w not in vocab:
                raise ValueError(f"Word '{w}' not in vocabulary")
            words.append(w)
            ids.append(vocab[w])

    ids += [vocab["[PAD]"]] * (max_len - len(ids))
    return words, ids

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

user_input = input("Enter a sentence with [MASK]: ")

words, encoded = encode_user_sentence(user_input)
input_tensor = torch.tensor([encoded]).to(device)

mask_position = (input_tensor == vocab["[MASK]"]).nonzero(as_tuple=True)

with torch.no_grad():
    output = model(input_tensor)
    masked_output = output[mask_position[0], mask_position[1]]
    prediction = torch.argmax(masked_output, dim=1)

predicted_word = inv_vocab[prediction.item()]

reconstructed = [
    predicted_word if w == "[MASK]" else w
    for w in words
]

print()
print("Input Sentence      :", " ".join(words))
print("Predicted Word     :", predicted_word)
print("Reconstructed Text :", " ".join(reconstructed))
