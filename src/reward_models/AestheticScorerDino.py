import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn

class AestheticScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        for param in self.dino.parameters():
            param.requires_grad = False
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),  # Example dimension
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1)
        )
    
    def forward(self, pixel_values):
        outputs = self.dino(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # Example of using the first token's representation
        return self.regressor(pooled_output)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

if __name__ == '__main__':
    pil_test = Image.open('data/raw/210.png').crop((0, 0, 512, 512))
    pixel_values = processor(images=pil_test, return_tensors="pt", padding=True)['pixel_values'].to('cuda')
    model = AestheticScorer().to('cuda')
    scores = model(pixel_values)
    breakpoint()
