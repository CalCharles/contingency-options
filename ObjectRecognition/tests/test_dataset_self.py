import matplotlib.pyplot as plt
from ObjectRecognition.dataset import DatasetSelfBreakout


bo_game = DatasetSelfBreakout('SelfBreakout/runs', 'SelfBreakout/runs/0', block_size=5)
INIT = 321
imgs =  bo_game.get_frame(INIT, INIT+12)
for i in range(12):
    plt.subplot(3, 4, i+1)
    # img = bo_game.get_frame(i+INIT)
    img = imgs[i]
    plt.imshow(img[0, ...])
plt.show()