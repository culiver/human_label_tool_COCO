import cv2
import numpy as np
import json
import os


limb = np.array([[0,1],[0,14],[0,15],[14,16],[15,17],[1,2],[1,5],[1,8],[1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13]])

targetVideo = 'video_lab_5G_220_08_run_train.avi'

out_dir = os.path.join('result', os.path.basename(targetVideo).split('.')[0])
os.makedirs(out_dir, exist_ok=True)

targetFrames = range(500, 700)

for pic_idx in targetFrames:
    print(pic_idx)

    imageToProcess = cv2.VideoCapture(targetVideo)

    imageToProcess.set(1, pic_idx - 1)
    rval, image = imageToProcess.read()
    image = cv2.resize(image,(800, 600))

    # Create a function based on a CV2 Event (Left button click)
    drawing = False  # True if mouse is pressed
    index = 0

    ix = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    iy = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    # mouse callback function
    def draw_polylines(event, x, y, flags, param):
        global ix, iy, drawing, mode, index, type

        if event == cv2.EVENT_RBUTTONDOWN:
             # When you click DOWN with left mouse button drawing is set to True
            drawing = True
            # Then we take note of where that mouse was located
            ix[index], iy[index] = 0, 0
            print("ix, iy: {}, {}".format(ix[index], iy[index]))
            index = index + 1

        if event == cv2.EVENT_LBUTTONDOWN:
            # When you click DOWN with left mouse button drawing is set to True
            drawing = True
            # Then we take note of where that mouse was located
            ix[index], iy[index] = x, y
            print("ix, iy: {}, {}".format(ix[index], iy[index]))
            index = index + 1

    # This names the window so we can reference it
    cv2.namedWindow('label', cv2.WINDOW_NORMAL)
    # Connects the mouse button to our callback function
    cv2.setMouseCallback('label', draw_polylines)

    while True:  # Runs forever until we break with Esc key on keyboard
        # Shows the image window
        cv2.imshow('label', image)
        # EXPLANATION FOR THIS LINE OF CODE:
        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163

        # CHECK TO SEE IF ESC WAS PRESSED ON KEYBOARD
        if (cv2.waitKey(1) & 0xFF == 27) or (ix==-1).sum()==0:
            break
    # Once script is done, its usually good practice to call this line
    # It closes all windows (just in case you have multiple windows called)
    cv2.destroyAllWindows()

    pose = np.stack((ix[0:], iy[0:]), axis=-1)
    print(pose)
    with open(os.path.join(out_dir,  '{}.json'.format(str(pic_idx))), 'w') as jsonfile:
        json.dump(pose.tolist(), jsonfile)

    # Draw the BBOX on the generated picture (picture_bbox_show)
    pts = np.stack((ix[0:], iy[0:]), axis=-1)
    pts.reshape((-1, 1, 2))
    print(pts)
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

    cv2.namedWindow('img_with_bbox', cv2.WINDOW_NORMAL)
    cv2.imshow("img_with_bbox", image)
    cv2.waitKey(0)

    # cv2.imwrite(PIC_BBOX_SHOW_DST + str(pic_idx) + ".jpg", image)

