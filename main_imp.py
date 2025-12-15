import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# Custom classes
from unet_arch import build_unet
from segment_ds import SegmentationDataset
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
])
# /home/champion/Documents/.Codes/Python/AMB82-Mini/Fish-eye/quantification/data-preparation-video/vid11/corner_mask"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_unet().to(device)

# def load_model_for_inference(checkpoint_path, model, device='cpu'):
    # Load the checkpoint
checkpoint_path = 'model/unet_checkpoint_arch_state.pth'#best unet_model
# checkpoint_path = 'unet_checkpoint_arch-mix_state.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model state dictionary
model.load_state_dict(checkpoint['model_state_dict'])
# Set model to evaluation mode for inference
model.eval()
# Move model to the specified device
model.to(device)
# Optionally, retrieve other saved information
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
# cap = cv2.VideoCapture('/home/champion/Documents/.Codes/Python/AMB82-Mini/videos/img_extraction/test3(170).mp4')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('Frame', frame)

# Efficient net
model_efficient = models.efficientnet_b0(pretrained=False)

num_features = model_efficient.classifier[1].in_features
model_efficient.classifier[1] = torch.nn.Linear(num_features, 5)

model_efficient.load_state_dict(torch.load("model/efficientnet_finetuned2.pth", map_location="cpu"))
model_efficient.eval()

captures_dir = f'camera-captures/images/{datetime.now().date()}'
os.makedirs(captures_dir, exist_ok=True)
counter = 0
cloud_data = {
    "cloud_coverage": 0,
    "cloud_types": []
}
time_data = {
    "date": '',
    "time": ''
}
cloud_types = {
    "A-clear sky": 0.0,
    "B-pattern": 0.0,
    "C-thick-dark": 0.0,
    "D-thick-white": 0.0,
    "E-veil": 0.0
}
@app.post('/upload_frame')
async def getImage(request: Request):
    global counter
    img = await request.body()
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filepath = os.path.join(captures_dir, f'image_{counter:04d}.jpg')
    cv2.imwrite(filepath, img)

    inference_dataset = SegmentationDataset(
        image_dir= captures_dir,
        mask_dir= None,
        transform=transform
    )

    test_loader = DataLoader(inference_dataset, batch_size=1, shuffle=True)
    # corner_mask = 'corner_mask'
    transformation = 'transformation'
    predictions = 'predictions'

    with torch.no_grad():
        sample_img = next(iter(test_loader))
        img = sample_img[0]
        img = img.permute(1, 2, 0)
        img = (img.cpu().numpy() * 255).astype('uint8')
        os.makedirs(transformation, exist_ok=True)
        transpath = os.path.join(predictions, f'{counter:03d}_transformation.png')
        cv2.imwrite(transpath, img)
        cv2.imshow('Transformed image', img)
        
        sample_img = sample_img.to(device)
        pred_mask = torch.sigmoid(model(sample_img))
        pred_mask = (pred_mask > 0.9).float()   

    mask_np = pred_mask[0].squeeze().cpu().numpy()  # Shape (H, W)

    # Convert to 0–255 uint8 for saving
    mask_img = (mask_np * 255).astype(np.uint8)
    pred_path = os.path.join(predictions, f'{counter:03d}_predicted_mask.png')
    cv2.imshow('Predictions', mask_img)
    # Save as image
    cv2.imwrite(pred_path, mask_img)

    total_pixels = cv2.countNonZero(mask_img)
    cloud_only = mask_img
    cloud_pixels = cv2.countNonZero(cloud_only)
    # height, width = mask_img.shape
    if total_pixels != 0:
        cloud_coverage = (cloud_pixels / (total_pixels)) * 8
        cloud_data['cloud_coverage'] = cloud_coverage
        print('Cloud coverage: ', cloud_coverage)

    print(f" Mask saved as {pred_path}")
    if img is None:
        print("No image found")
        # return {"status": "error", "message": "Failed to decode image"}
    counter += 1
    return cloud_data


@app.post('/get_cloud_types')
async def get_cloud_types(file: UploadFile = File(...)):
    frame = await file.read()
    nparr = np.frombuffer(frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    if frame is None:
        return JSONResponse({"error": "Failed to decode image"}, status_code=400)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))

    frame_pil = Image.fromarray(frame)

    input_tensor = train_transforms(frame_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model_efficient(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        probs = probs.detach().cpu().numpy().flatten()
    class_names = ["A-sky", "B-pattern", "C-thick-dark", "D-thick-white", "E-veil"]
    cloud_types = {name: float(prob) for name, prob in zip(class_names, probs)}

        # cloud_types['A-clear sky'], cloud_types['B-pattern'], cloud_types['C-thick-dark'], cloud_types['D-thick-white'], cloud_types['E-veil'] = outputs


    # class_names = ['A-sky', 'B-pattern', 'C-thick-dark', 'D-thick-white', 'E-veil']
    print("Model output", outputs.shape)
    print("Predicted class", predicted_class)
    print("Number of classes", len(class_names))
    # num_classes = len(class_names)
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    print(f"Predicted class: {class_names[predicted_class]}")
    print("Class probabilities:", probs)
    # cv2.imshow('frame', frame)

    return cloud_types, class_names[predicted_class]
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


# cap.release()
# cv2.destroyAllWindows()


@app.get('/get_time')
async def get_time():
    current_date = datetime.now().date()
    currenttime = datetime.now().time()
    time_data['date'] = current_date
    time_data['time'] = currenttime
    return time_data

# cloud_mask = 1 - pred_mask

    # Compute coverage for each image in the batch
    # cloud_pixels = pred_mask.sum(dim=[1, 2, 3])# total cloud pixels per image
    # total_pixels = torch.numel(pred_mask[0])# total pixels in one image
    # coverage_percent = (cloud_pixels / total_pixels) * 8

if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)