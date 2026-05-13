import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from models.densenet_crnn import DenseNetCRNN
from utils.dataset import OCRDataset
from utils.encode import encode, CHARS

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((96, 1024)),
    transforms.ToTensor()
])

dataset = OCRDataset("data/", transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = DenseNetCRNN(num_classes=len(CHARS)+1).to(device)

model.load_state_dict(torch.load("model.pth"))

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, texts in loader:
        images = images.to(device)

        targets = [encode(t) for t in texts]
        target_lengths = torch.tensor(
            [len(t) for t in targets],
            dtype=torch.long
        ).to(device)

        targets = torch.cat(targets).to(device)

        outputs = model(images)  # (T, B, C)

        input_lengths = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.long
        ).to(device)

        # print("Output length:", outputs.size(0))
        # print("Max target length:", max(target_lengths).item())

        loss = criterion(
            outputs.log_softmax(2),
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")