import cv2
import numpy as np
import matplotlib.pyplot as plt



limb = np.array([[0,1],[0,14],[0,15],[14,16],[15,17],[1,2],[1,5],[1,8],[1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13]])

targetVideo = 'video_lab_5G_220_08_run_train.avi'
TXT_DST = 'label/08_run_'

targetFrames = range(500, 700)

for pic_idx in targetFrames:
    print(pic_idx)

    imageToProcess = cv2.VideoCapture(targetVideo)
    imageToProcess.set(1, pic_idx - 1)
    rval, image = imageToProcess.read()




    # Write the BBOX coordinate to txt file
    fp = open(TXT_DST + str(pic_idx) + ".txt", "r")
    lines = fp.readlines()
    fp.close()
    poseVector_x = []
    poseVector_y = []

    for i in range(18):
        poseVector_x.append(float(lines[i].split(' ')[0]))
        poseVector_y.append(float(lines[i].split(' ')[-1].strip()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    poseVector_x = np.array(poseVector_x)
    poseVector_y = np.array(poseVector_y)
    for i in range(len(limb)):
        plt.plot(poseVector_x[[limb[i, 0], limb[i, 1]]], poseVector_y[[limb[i, 0], limb[i, 1]]], linewidth=3)
    # Draw the BBOX on the generated picture (picture_bbox_show)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    plt.close()


    # cv2.imwrite(PIC_BBOX_SHOW_DST + str(pic_idx) + ".jpg", image)

