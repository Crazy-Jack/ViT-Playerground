import json
from PIL import Image

import torch
import numpy as np 
from torchvision import transforms
import torchvision 
from pytorch_pretrained_vit import ViT
import PIL
from tqdm import tqdm
import time 
model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True).cuda()

# print(model.transformer.blocks[0])

# img = Image.open('img.jpg')
# # Preprocess image
# tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
# img = tfms(img).unsqueeze(0)

# Load class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# # Classify
# model.eval()
# with torch.no_grad():
#     outputs = model(img).squeeze(0)
# print('outputs-----')
# print(outputs.shape)
# print(outputs.max())
# for idx in torch.topk(outputs, k=3).indices.tolist():
#     print('[{idx}] {label:<75} ({p:.6f})'.format(idx=idx, label=labels_map[idx], p=outputs[idx].item()))
#     prob = torch.softmax(outputs, -1)[idx].item()
#     print('[{idx}] {label:<75} ({p:.6f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))


# find a set of image net training data with labels 

traindir="/user_data/tianqinl/imgnet/ILSVRC/Data/CLS-LOC/train"

normalize = transforms.Normalize(0.5, 0.5)

train_dataset = torchvision.datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize(model.image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(model.image_size),
        transforms.ToTensor(),
        normalize,
    ]))

batch_size = 1
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)
column = 255
head = 0
print("Dataloader built")
counter = 0
total_image = 1000
samples_per_image = 100
perturb_ratio = []
image_perturb_results = {}
pred_match_labels = []
for idx, (image, label) in tqdm(enumerate(train_loader), total=total_image):
    image_perturb_results['img_'+str(idx)] = {}
    model.eval()
    image = image.cuda()
    # original 
    with torch.no_grad():
        outputs_origin = torch.softmax(model(image, seed=0, which_column=column, which_head=head, save_attention=f"Normal_img_{idx}_column_{column}_head_{head}.jpeg"), -1).squeeze(0)
    pred = outputs_origin.argmax()
    match_label = (pred == label.item()).cpu()
    pred_match_labels.append(match_label)
    outputs_origin = outputs_origin[label.item()].cpu().item()
    image_perturb_results['img_'+str(idx)]['origin'] = outputs_origin 

    perturb_results = []
    perturb_pp_num = 0
    for perturb_id in range(samples_per_image):
        with torch.no_grad():
            outputs = torch.softmax(model(image, seed=int(time.time() * 10000), which_column=column, which_head=head), -1)
            # print(outputs.shape)
            outputs = outputs.squeeze(0)
            # print(outputs.shape)
            # print(f"label {label}")
            
            value_y = outputs[label.item()].cpu().item()
            ratio = np.exp(value_y) / np.exp(outputs_origin)
            perturb_results.append(ratio)
            if ratio < 1:
                perturb_pp_num += 1
    perturb_pp_num = perturb_pp_num / samples_per_image
    image_perturb_results['img_'+str(idx)]['perturbed'] = perturb_results
    print(f"Image {idx} perturb ratio {perturb_pp_num} pred_match_labels {match_label}")
    perturb_ratio.append(perturb_pp_num)
    counter += 1


    # if perturb_pp_num is 0
    if perturb_pp_num == 0:
        # it means no pertubation leads to worse outcome
        # can you show the attention map there? 
        print(f"print attend for img {idx}")
        with torch.no_grad():
            model(image, seed=0, which_column=column, which_head=head, save_attention=f"img_{idx}_column_{column}_head_{head}.jpeg")
        print(f"print img")
        print(image.shape)
        print(f"image.squeeze(0) {image.squeeze(0).shape}")
        print(f"image -- range {image.min()}, {image.max()}")
        image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
        image_np = image_np.astype(np.uint8)
        im = Image.fromarray(image_np)
        im.save(f"img_idx{idx}.png")
    
    
    if (counter % 50 == 0) or counter == total_image:
        np.save(f"pred_match_labels_c_{column}.npy", pred_match_labels)
        np.save(f"perturb_ratio_c_{column}.npy", perturb_ratio)
    
    if counter > total_image:
        break

# for i in image_perturb_results:
#     print(image_perturb_results[i])

# print(image_perturb_results)



# # ---------
# model.eval()
# img, label = next(iter(train_loader))

# with torch.no_grad():
#     outputs = model(img.cuda(), seed=torch.randint(100,size=(1,)).item())

# print(f"outputs {outputs.shape}")


# for i in range(batch_size):
#     print(f"Data {i} ----------- label {label} ")
#     for idx in torch.topk(outputs[i], k=3).indices.tolist():
        
#         prob = torch.softmax(outputs[i], -1)[idx].item()
#         print('[{idx}] {label:<5} ({p:.6f}%)'.format(idx=idx, label=idx, p=prob*100))

# for each data, systematically perturb each attention score with 100 seed 

