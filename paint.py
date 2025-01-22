import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

from project_cords import *

pygame.init()
matplotlib.rcParams['font.family'] = 'HanaMinA'

W = H = 106*6

win = pygame.display.set_mode((W, H))
pygame.display.set_caption("Draw Kanji")

pygame.draw.line(win, (20, 20, 20), (0, W/2), (H, W/2), 3)
pygame.draw.line(win, (20, 20, 20), (H/2, 0), (H/2, W), 3)

def main():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        if mouse_pressed[0]:
            pygame.draw.circle(win, (255, 255, 255), mouse_pos, 9)

        pygame.display.flip()

    img = pygame.surfarray.pixels3d(win)

    pygame.quit()
    
    return np.transpose(255-img, (1, 0, 2))

if __name__ == "__main__":
    orig = main()
    
    device = torch.device('cuda:0')
    net: Net = torch.load("models/model.pth", weights_only=False)

    img = TRANSFORM(orig)
    img = img.reshape((1, 1, 100, 100))
    
    fig, ax = plt.subplots()
    ax.imshow(img[0, 0], cmap="gray", extent=[0, 106, 106, 0])
    
    img = img.to(device)
    result:torch.Tensor = net(img)

    for ind, result in enumerate(np.split(result.cpu().detach().numpy()[0, :], len(TOP_RADICALS))):
        print(TOP_RADICALS[ind], np.round(result, 1))
        
        if all(result > 2):
            rect = patches.Rectangle((result[0], result[1]), result[2]-result[0], result[3]-result[1], linewidth=2, edgecolor='r', facecolor='r', alpha=.6)
            ax.text(result[0], result[1]-3, TOP_RADICALS[ind], color='green', fontsize=24, ha='center', va='center')
            ax.add_patch(rect)

    plt.show()