import cv2
import os, sys
import imageio as imio

if __name__ == '__main__':
    pth, folder, start, num, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    print(pth + "/" + str(folder) + "state" + str(start) + ".png")
    im = cv2.imread(os.path.join(pth, str(folder), "state" + str(start) + ".png"))
    height, width, layers = im.shape
    end = num+start
    for i in range(start,end):
        im = cv2.imread(os.path.join(pth,str(folder + i // 2000),"state" + str(i % 2000) + ".png"))
        print(os.path.join(pth,str(folder + i // 2000),"state" + str(i % 2000) + ".png"))
        if im is not None:
            cv2.imshow('frame',im)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
