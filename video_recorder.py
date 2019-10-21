import cv2
import os
import imageio as imio
from ChangepointDetection.CHAMP import CHAMPDetector




start = 0
fps = 40
end = 1000
pth = "data/pusherrandom/0/"
# video_name = "demopushervid.mp4"
# pth = "data/goodgripper/0/"
top_pth = "data/goodgripper/"
video_name = "pusherrandom.mp4"
# pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# top_pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# video_name = "ballintensity.mp4"

prefix = "state"
# prefix = "intensity_"
# prefix = "marker_"
# prefix = "focus_img_"

changepoint_flashes = False

class dummyargs():
    def __init__(self):
        self.record_rollouts = top_pth
        self.num_iters = -1

args = dummyargs()

if changepoint_flashes:
    train_edge = "Paddle->Ball"
    paddle_params = [3, 100, 1, 100, 100, 2, 1e-2, 3]
    ball_params = [15, 10, 1, 100, 100, 2, 1e-2, 3] 
    detector = CHAMPDetector(train_edge, ball_params)
    data = detector.load_obj_dumps(args, dumps_name='object_dumps.txt')
    models, changepoints = detector.generate_changepoints(data, save_dict=False)

im = cv2.imread(pth + prefix + str(start) + ".png")
print(im, pth + prefix + str(start) + ".png")
height, width, layers = im.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))
for i in range(start,end):
    im = cv2.imread(pth + prefix + str(i) + ".png")
    if changepoint_flashes and i in changepoints:
        im = cv2.bitwise_not(im)
    if im is not None:
        video.write(im)
        print(im)
        cv2.imshow('frame',im)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
video.release()
