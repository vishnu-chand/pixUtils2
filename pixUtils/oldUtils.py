# from .pixCommon import *
#
#
# def bbox2cent(bbox):
#     '''
#     calculate the centroid of bbox
#     '''
#     x, y, w, h = bbox
#     return np.array((x + w / 2, y + h / 2))
#
#
# def img2bbox(img):
#     return np.array([0, 0, img.shape[1], img.shape[0]])
#
#
# def bbox2dbbox(img, bbox):
#     '''
#     convert cv bbox to dlib bbox
#     '''
#     x, y, w, h = bbox
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     return dlib.rectangle(x, y, x + w, y + h)
#
#
# def dbbox2bbox(img, dbbox):
#     '''
#     convert dlib bbox to cv bbox
#     '''
#     x = dbbox.left()
#     y = dbbox.top()
#     w = dbbox.right() - x
#     h = dbbox.bottom() - y
#     return frameFit(img, (x, y, w, h))
#
#
# def getTrajectory(p1, p2, returnSemiCircle=False):
#     '''
#     when returnSemiCircle is true  it will return in range -180 to 180
#     when returnSemiCircle is false it will return in range 0 to 360
#             self.matrix, self.chessImgW, self.chessImgH = None, None, None
#
#     '''
#     delta = np.array(p1) - np.array(p2)
#     theta = cv2.fastAtan2(delta[1], delta[0])
#     magnitude = cv2.norm(np.array(delta))
#     if returnSemiCircle:
#         if theta > 180:
#             theta -= 360
#     return np.array((magnitude, theta))
#
#
# def skeletonIt(img):
#     skel = np.zeros_like(img)
#     ret, img = cv2.threshold(img, 127, 255, 0)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     done = False
#
#     while not done:
#         eroded = cv2.erode(img, element)
#         temp = cv2.dilate(eroded, element)
#         temp = cv2.subtract(img, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img = eroded.copy()
#
#         zeros = img.size - cv2.countNonZero(img)
#         if zeros == img.size:
#             done = True
#     return skel
#
#
# def getOverlap(bbox1, bbox2, reference='iou'):
#     x1min, y1min, x1max, y1max = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
#     x2min, y2min, x2max, y2max = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
#     width_of_overlap_area = min(x1max, x2max) - max(x1min, x2min)
#     height_of_overlap_area = min(y1max, y2max) - max(y1min, y2min)
#     score = 0.0
#     if width_of_overlap_area > 0 and height_of_overlap_area > 0:
#         area_of_overlap = width_of_overlap_area * height_of_overlap_area
#         if reference == 'iou':
#             box_1_area = (y1max - y1min) * (x1max - x1min)
#             box_2_area = (y2max - y2min) * (x2max - x2min)
#             denominator = box_1_area + box_2_area - area_of_overlap
#         elif reference == 'minBbox':
#             bbox1Area = bbox1[2] * bbox1[2]
#             bbox2Area = bbox2[2] * bbox2[2]
#             denominator = bbox1Area if bbox1Area < bbox2Area else bbox2Area
#         elif reference == 'maxBbox':
#             bbox1Area = bbox1[2] * bbox1[2]
#             bbox2Area = bbox2[2] * bbox2[2]
#             denominator = bbox1Area if bbox1Area > bbox2Area else bbox2Area
#         else:  # a reference bbox has been passed
#             denominator = reference[2] * reference[3]
#         score = area_of_overlap / denominator
#     return score
#
#
# class ImLog:
#     def __init__(self):
#         self.__imgs = []
#         self.__rcsize = None
#         self.size = 0
#
#     def log(self, name, img, loc=(30, 30), color=(255, 255, 255), txtSize=1, txtFont=cv2.FONT_HERSHEY_SIMPLEX, txtThickness=3, txtColor=None):
#         self.size += 1
#         img = bboxLabel(img.copy(), name, loc, color, 3, txtSize, txtFont, txtThickness, txtColor)
#         self.__imgs.append(img)
#         return self
#
#     def reset(self):
#         self.__imgs = []
#         self.__rcsize = None
#         self.size = 0
#
#     def setSize(self, rcsize):
#         self.__rcsize = rcsize[:2]
#
#     def getImg(self, rcsize=None, nCol=None, resize_method=cv2.INTER_LINEAR, fit=False, asgray=False):
#         if rcsize is not None:
#             self.__rcsize = rcsize
#         if self.__rcsize is None:  # take the first image size
#             self.__rcsize = self.__imgs[0].shape[:2]
#         img = photoframe(self.__imgs, self.__rcsize, nCol, resize_method, fit, asgray)
#         self.reset()
#         return img
#
#
# def simpleTrackBar(readOnlyInputImg, trackBarsLengths, winname='trackBar'):
#     cv2.namedWindow(winname, 0)
#     for ix, maxValue in enumerate(trackBarsLengths):
#         cv2.createTrackbar(str(ix), winname, 0, maxValue, lambda x: None)
#     outputImg = readOnlyInputImg.copy()
#     while True:
#         cv2.imshow(winname, outputImg)
#         k = cv2.waitKey(100) & 0xFF
#         if k == 27:
#             break
#         vals = [readOnlyInputImg, outputImg]
#         for ix, _ in enumerate(trackBarsLengths):
#             vals.append(cv2.getTrackbarPos(str(ix), winname))
#         yield vals
#
#
# class __MarkRoi:
#     def __init__(self, oimg, color):
#         self.contours = []
#         self.oimg = oimg.copy()
#         self.dispImg = self.oimg.copy()
#         self.color = color
#
#     def reset(self):
#         self.contours = []
#         self.dispImg = self.oimg.copy()
#
#     def drawRoi(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDBLCLK:
#             self.dispImg = self.oimg.copy()
#             self.contours.append((x, y))
#             contour = np.array(self.contours).astype(np.int32)
#             cv2.drawContours(self.dispImg, [contour], -1, self.color, 10)
#
#     def markRoi(self, key=ord('m'), winname='mark roi by double click'):
#         res = None
#         if key == ord('m'):
#             cv2.namedWindow(winname, 0)
#             cv2.setMouseCallback(winname, self.drawRoi)
#             while 1:
#                 cv2.imshow(winname, self.dispImg)
#                 k = cv2.waitKey(1) & 0xFF
#                 if k == 13:
#                     break
#                 if k == 27:
#                     self.reset()
#             val = np.array([np.array(self.contours).astype(np.int32)])
#             if val.any():
#                 res = val
#         cv2.destroyAllWindows()
#         return res
#
#
# def markRoi(oimg, key=ord('m'), winname='mark roi by double click', color=(255, 0, 0)):
#     return __MarkRoi(oimg, color).markRoi(key, winname)
#
#
# def readYaml(src, defaultDict=None):
#     data = defaultDict
#     if os.path.exists(src):
#         with open(src, 'r') as book:
#             data = yaml.safe_load(book)
#     return DotDict(data)
#
#
# def writeYaml(yamlPath, jObjs):
#     with open(yamlPath, 'w') as book:
#         yaml.dump(yaml.safe_load(jObjs), book, default_flow_style=False, sort_keys=False)
#
#
# def checkAttr(obj, b, getAttr=False):
#     a = set(vars(obj).keys())
#     if getAttr:
#         print(a)
#     extra = a - a.intersection(b)
#     if len(extra):
#         raise Exception(extra)
#
#
# def drawText(img, txt, loc, color=(255, 255, 255), txtSize=1, txtFont=cv2.FONT_HERSHEY_SIMPLEX, txtThickness=3, txtColor=None):
#     (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
#     x0, y0 = int(loc[0]), int(loc[1])
#     if txtColor is None:
#         txtColor = (0, 0, 0)
#     cv2.rectangle(img, (x0, y0), (x0 + w, y0 - h - baseLine), color, -1)
#     cv2.putText(img, txt, (x0, y0 - baseLine), txtFont, txtSize, txtColor, txtThickness)
#     return img
#
#
# def maskIt(roi, roiMask):
#     """
#     apply mask on the image. It can accept both gray and colors image
#     """
#     if len(roi.shape) == 3 and len(roiMask.shape) == 2:
#         roiMask = cv2.cvtColor(roiMask, cv2.COLOR_GRAY2BGR)
#     elif len(roi.shape) == 2 and len(roiMask.shape) == 3:
#         roiMask = cv2.cvtColor(roiMask, cv2.COLOR_BGR2GRAY)
#     return cv2.bitwise_and(roi, roiMask)
#
#
# def imHconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
#     rh, rw = sizeRC[:2]
#     res = []
#     for queryImg in imgs:
#         qh, qw = queryImg.shape[:2]
#         queryImg = cv2.resize(queryImg, (int(rw * qw / qh), int(rh)), interpolation=interpolation)
#         res.append(queryImg)
#     return cv2.hconcat(res)
#
#
# def imVconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
#     rh, rw = sizeRC[:2]
#     res = []
#     for queryImg in imgs:
#         qh, qw = queryImg.shape[:2]
#         queryImg = cv2.resize(queryImg, (int(rw), int(rh * qh / qw)), interpolation=interpolation)
#         res.append(queryImg)
#     return cv2.vconcat(res)
#
#
# def getSubPlots(nrows=1, ncols=-1, axisOff=True, tight=True, figsize=(15, 15), sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
#     if ncols != -1:
#         nimgs = nrows * ncols
#     else:
#         nimgs = nrows
#         nrows = int(np.ceil(nimgs ** .5))
#         ncols = int(nimgs / nrows)
#         ncols = ncols + 1 if (nrows * ncols) < nimgs else ncols
#
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)
#
#     if nrows == 1 and ncols == 1:
#         axs = np.array([axs])
#     axs = axs.ravel()
#     for ix, ax in enumerate(axs):
#         if ix < nimgs:
#             yield lambda x: ax.set_title(x, color='white'), ax
#         if axisOff:
#             ax.axis('off')
#     if tight:
#         plt.tight_layout()
#     if axisOff:
#         plt.axis('off')
# # --------------------------------------- jupyter and colab utils --------------------------------------- #
# def unwantedThings():
#     # ################################################ clear gdrive trash #################################################
#     # !df -h
#     # !rm -rf ~/.local/share/Trash/*
#     # !df -h
#
#     # from google.colab import auth
#     # from tqdm.notebook import tqdm
#     # from pydrive.auth import GoogleAuth
#     # from pydrive.drive import GoogleDrive
#     # from oauth2client.client import GoogleCredentials
#
#     # auth.authenticate_user()
#     # gauth = GoogleAuth()
#     # gauth.credentials = GoogleCredentials.get_application_default()
#     # my_drive = GoogleDrive(gauth)
#
#     # def deleteDriveTrash(book=''):
#     #     query = "trashed=true"
#     #     if book:
#     #         query = f"title = '{book}' and trashed=true"
#     #     for a_file in tqdm(my_drive.ListFile({'q': query}).GetList()):
#     #         # print(f'the file {a_file["title"]}, is about to get deleted permanently.')
#     #         try:
#     #             a_file.Delete()
#     #         except:
#     #             pass
#
#     # deleteDriveTrash()
#     # ################################################ clear gdrive trash #################################################
#
#     # ############################################# load 2nd drive #############################################
#     # !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
#     # !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
#     # !apt-get update -qq 2>&1 > /dev/null
#     # !apt-get -y install -qq google-drive-ocamlfuse fuse
#     # from google.colab import auth
#     # auth.authenticate_user()
#     # from oauth2client.client import GoogleCredentials
#     # creds = GoogleCredentials.get_application_default()
#     # import getpass
#     # !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
#     # vcode = getpass.getpass()
#     # !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
#     # !mkdir -p /content/drive2
#     # !google-drive-ocamlfuse /content/drive2
#     # ############################################# load 2nd drive #############################################
#
#     # !git clone https://github.com/tensorflow/models.git
#     # dirop('/content/drive/My Drive/research/deeplab/trainBG', mv='/content/drive/My Drive/trainBG')
#     # dirop('/content/drive/My Drive/research', rm=True)
#     # dirop('/content/models/research/slim', mv='/content/drive/My Drive/research/slim')
#     # dirop('/content/models/research/deeplab', mv='/content/drive/My Drive/research/deeplab')
#     # dirop('/content/drive/My Drive/trainBG', mv='/content/drive/My Drive/research/deeplab/trainBG')
#     # !rm -rf '/content/models'
#
#     '''
#     https://github.com/ZHKKKe/MODNet
#     https://github.com/zhanghang1989/ResNeSt
#     https://paperswithcode.com/paper/resnest-split-attention-networks
#
#     !cd /content;git clone https://github.com/thuyngch/Human-Segmentation-PyTorch.git
#
#     # https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019
#     # https://storage.googleapis.com/kaggle-data-sets/358221/702372/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201128%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201128T100831Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=0438f9d4ec37fa75a8467b1a62e48babcc32b7d9e3ba397b0ae29e74e0e2406b548c29ec811d5a1ee6e2b3fa495151301090b804bdd855e4ffcd48948c70a66af2e51eda79f9b7c60bb373f70f6e37dda787acad35a0910a868a818611c85a428ccc6eac1ce53f89b440fad54f64fa88e414a90f50eed78578ba6d358c58ed8d9f58a4e790d0b02b3393043b537b9a3c2bc804217c9eb42ff1442d48160125c0670c61aee1fc0f24bd66c0c713134c63c775aec1d789beac106b620d510bf019a18645abd8e3495ac4f6f05b963a48bcb3cff7b126dfb901710f9756d0dd349ad25940b78bece8c3bd614a5f7d46c4c825d51c5c7190935c5fb3383f18b01bf5
#     https://zenodo.org/record/2654485/files/Indoor%20Object%20Detection%20Dataset.zip?download=1
#
#     https://github.com/dong-x16/PortraitNet
#     https://github.com/lizhengwei1992/Fast_Portrait_Segmentation
#
#
#     https://colab.research.google.com/drive/10eGmnbXV-NVl-iSMwECwrSESob37V2kh#scrollTo=qjidyZ76WYeW
#     '''
#
#
# import tarfile
# import numpy as np
# import skimage.io as io
# from copy import deepcopy
# from pixUtils import *
#
# try:
#     from google.colab import files as coFile
#     from google.colab.patches import cv2_imshow
#     cv2.imshow = lambda winName, img: cv2_imshow(img)
# except:
#     pass
#
#
# def releaseTf():
#     try:
#         import tensorflow as tf
#         # device = cuda.get_current_device()
#         # device.reset()
#         tf.reset_default_graph()
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         sess = tf.Session(config=config)
#     except:
#         pass
#
#
# def reloadPy():
#     def _is_module_deletable(modname, modpath):
#         if modname.startswith('_cython_inline'):
#             # Don't return cached inline compiled .PYX files
#             return False
#         for path in [sys.prefix]:
#             if modpath.startswith(path):
#                 return False
#         else:
#             return set(modname.split('.'))
#
#     """
#     Del user modules to force Python to deeply reload them
#
#     Do not del modules which are considered as system modules, i.e.
#     modules installed in subdirectories of Python interpreter's binary
#     Do not del C modules
#     """
#     log = []
#     for modname, module in list(sys.modules.items()):
#         modpath = getattr(module, '__file__', None)
#
#         if modpath is None:
#             # *module* is a C module that is statically linked into the
#             # interpreter. There is no way to know its path, so we
#             # choose to ignore it.
#             continue
#
#         if modname == 'reloader':
#             # skip this module
#             continue
#
#         modules_to_delete = _is_module_deletable(modname, modpath)
#         if modules_to_delete:
#             log.append(modname)
#             del sys.modules[modname]
#
#     print(f"Reloaded modules: {log}")
#
#
# def quit():
#     raise Exception('stopping execution')
# # --------------------------------------- jupyter and colab utils --------------------------------------- #
