import imageio as imio
import cv2
import numpy as np

pth = "../datasets/caleb_data/blockvec/39/"

imv = np.zeros(imio.imread(pth + "state" + str(0) + ".png").shape)
imv += imio.imread(pth + "state" + str(690) + ".png") / 10
for i in range(690,710):
    im = imio.imread(pth + "state" + str(i) + ".png")
    cv2.imshow('frame',im)
    print(i)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    imv += im / 30
imio.imwrite("blur.png", imv)
