from scipy.io import loadmat, savemat
import cv2
import numpy as np
import matplotlib.pyplot as plt
# path = './lu177-Data/Whole_Data/density'
readpath = '/Volumes/Samsung_T5/Dropbox (University of Michigan)/research/jf_yuni/lu177patients/ga68petonly/vp22-pet/recon/density'
writepath = '/Volumes/Samsung_T5/Dropbox (University of Michigan)/research/jf_yuni/lu177patients/ga68petonly/vp22-pet/recon/'
for num in range(0, 1):
    print('Processing {} density map'.format(num + 1))
    # num_fill = str(num + 1).zfill(3)
    num_fill = '_pet22'
    x = loadmat(readpath + num_fill + '.mat')['x']
    z = x.copy()
    mask = np.zeros_like(x, dtype=bool)
    for it in range(x.shape[-1]):
        img = x[:,:,it]
        _, thresh = cv2.threshold(img, 800, 3500, cv2.THRESH_BINARY)
        # mask = np.zeros_like(img, dtype=bool)
        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(thresh, kernel, iterations=2)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(erode, kernel, iterations=3)
        dilate = np.uint8(dilate)
        contours, hierarchy = cv2.findContours(dilate , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(cntsSorted) != 0:
            max_contour = cntsSorted[0]

            for j in range(img.shape[0]):
                for i in range(img.shape[1]):
                    if cv2.pointPolygonTest(max_contour, (i, j), False) <= 0:
                        img[j, i] = 1
                    else:
                        mask[j, i, it] = 1
            # plt.figure()
            # plt.imshow(np.transpose(img))
            # plt.show()
            z[:,:,it] = img
    z = np.float32(z)
    savemat(readpath + '_rm' + num_fill + '.mat', {'x': z}, do_compression= True)
    savemat(writepath + 'mask' + num_fill + '.mat', {'x': mask}, do_compression= True)
