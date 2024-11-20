import pygame
import numpy as np
import torch
from project import *
from PIL import Image
# import matplotlib.pyplot as plt

pygame.init()

W = H = 106*6

win = pygame.display.set_mode((W, H))
pygame.display.set_caption("Draw Kanji")

# pygame.draw.line(win, (20, 20, 20), (0, W/2), (H, W/2), 3)
# pygame.draw.line(win, (20, 20, 20), (H/2, 0), (H/2, W), 3)

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

    return Image.fromarray(np.flip(np.rot90(np.uint8(img), k=-1), axis=1)).convert("RGB")

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net: Net = torch.load("models/model.pth", weights_only=False)
    
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
