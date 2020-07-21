import cv2
import io
from os.path import dirname, join
from android.os import Environment
import numpy as np
def test(o):
    filename = join(dirname(__file__), "sam.jpg")


    decoded = cv2.imdecode(np.frombuffer(o, np.uint8), -1)
    #src = cv2.imread(decoded)
    grayScale = cv2.cvtColor( decoded, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #cv2.imshow("BlackHat",blackhat)
    #cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    #print( thresh2.shape )
    #cv2.imshow("Thresholded Mask",thresh2)
    #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(decoded,thresh2,1,cv2.INPAINT_TELEA)
    #data= np.array(dst,dtype="int32")
    im_resize = cv2.resize(dst, (224, 224))

    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    byte_im = im_buf_arr.tobytes()
    d = str(Environment.getExternalStorageDirectory())
   # cv2.imwrite(d+'sample1.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    final= join(d,'sample1.jpg')

    return byte_im