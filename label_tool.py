import cv2
import numpy as np
import json
import os
import math

def draw_bodypose(canvas, keypoints, body_part='top'):
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(len(colors)):
        x, y = keypoints[i][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(len(limbSeq)):
        cur_canvas = canvas.copy()
        Y = keypoints[np.array(limbSeq[i]), 0]
        X = keypoints[np.array(limbSeq[i]), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

limbSeq = np.array([[0,1],[0,14],[0,15],[14,16],[15,17],[1,2],[1,5],[1,8],[1,11],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13]])

targetVideo = 'video_lab_5G_220_08_run_train.avi'

out_dir = os.path.join('result', os.path.basename(targetVideo).split('.')[0])
os.makedirs(out_dir, exist_ok=True)

targetFrames = (450, 700)
pic_idx = targetFrames[0]
# for pic_idx in targetFrames:
while pic_idx < targetFrames[1]:
    print(pic_idx)

    imageToProcess = cv2.VideoCapture(targetVideo)

    imageToProcess.set(1, pic_idx)
    rval, image = imageToProcess.read()
    image = cv2.resize(image,(800, 600))

    # Create a function based on a CV2 Event (Left button click)
    drawing = False  # True if mouse is pressed
    index = 0

    click_points = np.ones((18, 2))*-1

    # mouse callback function
    def draw_polylines(event, x, y, flags, param):
        global ix, iy, drawing, mode, index
        if event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            click_points[index]= np.array([0, 0])
            print("ix, iy: {}, {}".format(click_points[index,0], click_points[index,1]))
            index = index + 1
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            click_points[index]= np.array([x, y])
            print("ix, iy: {}, {}".format(click_points[index,0], click_points[index,1]))
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
        if (cv2.waitKey(1) & 0xFF == 27) or index==len(click_points):
            break
    # Once script is done, its usually good practice to call this line
    # It closes all windows (just in case you have multiple windows called)
    cv2.destroyAllWindows()

    keypoints = np.zeros((18, 3))
    for idx, useful_idx in enumerate(range(18)):    
        keypoints[useful_idx, :2] = click_points[idx]
        keypoints[useful_idx, 2] = 1
    keypoints = keypoints.astype(int)    
    print(keypoints)
    pose_format = {
            "version": 1.3,
            "people": [
                        {
                            "person_id": [-1],
                            "pose_keypoints_2d": keypoints.reshape(-1).tolist(),
                            "face_keypoints_2d": [],
                            "hand_left_keypoints_2d": [],
                            "hand_right_keypoints_2d": [],
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d": [],
                            "hand_right_keypoints_3d": []
                        }
                    ]
                }
    with open(os.path.join(out_dir,  '{}_keypoints.json'.format(str(pic_idx))), 'w') as jsonfile:
        json.dump(pose_format, jsonfile)

    skeleton = draw_bodypose(image, keypoints, body_part='lower')
    cv2.namedWindow('skeleton', cv2.WINDOW_NORMAL)
    cv2.imshow("skeleton", skeleton)
    # cv2.waitKey(0)
    decision = cv2.waitKey(0)
    # Backspace
    if decision == 8:
        continue
    else:
        pass
    pic_idx += 1

