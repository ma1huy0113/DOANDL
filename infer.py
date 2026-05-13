import torch
from PIL import Image
from torchvision import transforms

from models.densenet_crnn import DenseNetCRNN
from utils.encode import decode, CHARS

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DenseNetCRNN(len(CHARS)+1)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((96, 1024)),
    transforms.ToTensor()
])

def predict(image):
    image = image.convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    pred = output.argmax(2)[:, 0]
    return decode(pred.cpu().numpy())

img = Image.open(r"data/20140603_0003_BCCTC_tg_0_2.png").convert("RGB")

result = predict(img)

print("Prediction:")
print(result)