# Setup cell.
import time, os, json
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.transformer_layers import *
from utils.captioning_solver_transformer import CaptioningSolverTransformer
from utils.transformer import CaptioningTransformer
from utils.coco_utils import load_coco_data, decode_captions

torch.manual_seed(231)
np.random.seed(231)

data = load_coco_data(pca_features=False, base_dir='/Users/tt/Documents/Hanyang/3rd grade/Computer Vision/HW_3/NICE/datasets/coco_captioning')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Now using {device} device")

transformer = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=256,
          num_heads=4,
          num_layers=2,
          max_length=30
        ).to(device)

transformer_solver = CaptioningSolverTransformer(
           transformer, data, idx_to_word=data['idx_to_word'],
           num_epochs=20,
           batch_size=20,
           learning_rate=1e-3,
           verbose=True, print_every=1000,
         )

model_num = sys.argv[1]

try:
    print("Start training.")
    transformer_solver.train()
    transformer_solver.train(split='val')
    print("Finished training.")
except:
    print("Stopped training.")
torch.save(transformer.state_dict(), f'./models/trained_{model_num}.pt')

def generate_caption(feature):
    captions = transformer.sample(feature)
    captions = decode_captions(captions, data['idx_to_word'])
    return captions[0]

student_id = "2022094093"
pred = []
nice_feat = data['nice_feature']
nice_feat = np.expand_dims(nice_feat, axis=1)

for i in tqdm(range(len(nice_feat))):
    caption = generate_caption(nice_feat[i])
    image_id = i + 1
    pred.append({'image_id' : image_id, 'caption' : caption})

result = {"student_id" : student_id, "prediction" : pred}
json.dump(result, open(f'./prediction/prediction_{model_num}.json', 'w'), indent='\t')