import pygame
import numpy as np
import torch
from project import *
from PIL import Image

pygame.init()

win = pygame.display.set_mode((106*6, 106*6))
pygame.display.set_caption("Draw Kanji")

def main():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        if mouse_pressed[0]:
            pygame.draw.circle(win, (255, 255, 255), mouse_pos, 5)

        pygame.display.flip()

    img = pygame.surfarray.pixels3d(win)
    pygame.quit()

    return Image.fromarray(np.uint8(img)).convert("RGB")

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net: Net = torch.load("models/model_crazy.pth", weights_only=False)
    
    img = main()
    img = TRANSFORM(img)
    img = img.reshape((1, 1, 100, 100))
    img = img.to(device)
    
    net.to(device)
    net.eval()
    
    accuracy = {k: [] for k in TOP_RADICALS}
    result = net(img)

    for i, radical in enumerate(TOP_RADICALS):
        r = result[0][i].item()
        print(radical, round(r, 2))
