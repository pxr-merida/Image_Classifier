#Run Script with
#python predict.py flowers/test/43/image_02365.jpg checkpnt.pth --gpu
# Other Available options
#python predict.py --top_k 5



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import json
import argparse


def load_checkpoint(filename='checkpnt.pth'):
    models_list = {'densenet121': models.densenet121(pretrained=True),
                   'densenet161': models.densenet161(pretrained=True),}
    checkpoint = torch.load(filename)
    model = models_list[checkpoint['arch']]
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([('fc1',
                                             nn.Linear(1024,
                                                       checkpoint['hidden_units'],
                                                       bias=True)),
                                            ('Relu1', nn.ReLU()),
                                            ('Dropout1', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(
                                                checkpoint['hidden_units'],
                                                102, bias=True)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model


def process_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    image.thumbnail((1000000, 256)) if width > height else image.thumbnail(
        (256, 200000))
    width, height = image.size
    left = (width - 224) / 2.
    top = (height - 224) / 2.
    right = (width + 224) / 2.
    bottom = (height + 224) / 2.
    im = image.crop((left, top, right, bottom))
    im_arr = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_arr = (im_arr - mean) / std
    im_arr = im_arr.transpose((2, 0, 1))

    return im_arr


def predict(image_path, model, topk=5, category_names='cat_to_name.json',
            device='cuda'):

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    image = process_image(image_path)
    img = torch.from_numpy(image).type(torch.FloatTensor)
    image = img.unsqueeze(0)
    image = image.float().cuda()
    model.eval()
    model.to(device)
    with torch.no_grad():
        model = model.double()
        output = model.forward(image.double())
    ps = torch.exp(output)
    probs, classes = torch.topk(ps, topk)
    probs = probs.cpu().numpy().tolist()[0]
    classes = classes.cpu().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in classes]
    flowers = [cat_to_name[str(x)] for x in classes]

    return probs, flowers


def view_classify(image_path, probs, flowers):
    df = pd.DataFrame({'probability': probs, 'class': flowers})
    im = Image.open(image_path)
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.array(im))
    plt.axis('off')
    plt.subplot(2, 1, 2)
    sns.barplot(x="probabilty", y="class", data=df, orient="h")
    plt.tight_layout()
    fig.savefig('Fig1.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict flower class')
    parser.add_argument('image_path', help='image  path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--top_k', help='Top most classes')
    parser.add_argument('--category_names',
                        help='Mapping')
    parser.add_argument('--gpu', help='gpu on', action="store_true")
    args = parser.parse_args()
    topk = args.top_k if args.top_k else 5
    category_names = args.category_names if args.category_names else '/home/workspace/ImageClassifier/cat_to_name.json'
    device = 'cuda' if args.gpu else 'cpu'

    model = load_checkpoint(filename=args.checkpoint)
    probs, classname = predict(args.image_path, model, topk=topk,
                             category_names=category_names, device=device)
    df = pd.DataFrame({'probability': probs, 'flower': classname})
    df.to_csv('Prob_Classname.csv')