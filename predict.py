import os
import torch
from PIL import Image
from torchvision import transforms
from nets.swin_transformer import SwinTransformer
from nets.lyj_swin_transformer import LYJSwinTransformer


def predict_res(model_path, model_name, img_path):
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if model_name == "swin-t":
        model = SwinTransformer(num_classes=100).to(device)
    elif model_name == "lyj_swin-t":
        model = LYJSwinTransformer(num_classes=100).to(device)
    else:
        raise ValueError("model name must be swin-t!")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    label_names = sorted(os.listdir("ImageNet100/train"))
    image = torch.unsqueeze(transform(Image.open(img_path).convert("RGB")), dim=0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(image), dim=-1).cpu().numpy()[0]
    print(f"{img_path}的预测结果是:{label_names[pred]}")

    return label_names[pred]


if __name__ == '__main__':
    predict_res(
        model_path="models/lyj_swin-t_best.pth",
        model_name="lyj_swin-t",
        img_path="ImageNet100/test/Americanegret/n02009912_18676.JPEG"
    )






