import matplotlib.pyplot as plt
from functools import partial
from SelfBreakout.breakout_screen import RandomConsistentPolicy
from ObjectRecognition.dataset import DatasetAtari


game = DatasetAtari(
    'BreakoutNoFrameskip-v4',
    partial(RandomConsistentPolicy, change_prob=0.35),
    n_state=1000,
    save_path='results',
    binarize=0.1,
)

# plt.imshow(game.get_frame(5)[0]); plt.show()

PX, PY = 4, 5
FRAMES = PX * PY
LIMIT = 1000
for init in range(0, LIMIT, FRAMES):
    # imgs =  game.get_frame_batch(init, init+12)
    for i in range(FRAMES):
        ax = plt.subplot(PX, PY, i+1)
        img = game.get_frame(i+init)
        # img = imgs[i]
        ax.imshow(img[0, ...])
        ax.set_title('action: %d'%game.retrieve_action([i+init])[0])
        ax.axis('off')
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()