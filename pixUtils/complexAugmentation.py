from pixUtils import *
import albumentations as A
from skimage.filters import gaussian


def InterleaveAug(maxInterleaveHeightPc=0.03, maxInterleaveWidthPc=0.03, p=0.5):
    '''
    maxInterleaveHeightPc: (0.02, 0.15) if maxInterleaveHeightPc=.5, 50% of image will be replace with interleaveImg
    maxInterleaveWidthPc: (0.02, 0.15) if maxInterleaveWidthPc=.1, 10% of image will be replace with interleaveImg
    interleaveImg: image whose pixels will be used for interleaving
    '''

    def __prcs(image, interleaveImg, force_apply=False, **kw):
        if force_apply or random.random() < p:
            imH, imW = image.shape[:2]
            i = random.sample(range(imH), int(imH * maxInterleaveWidthPc))
            j = random.sample(range(imW), int(imW * maxInterleaveHeightPc))
            interleaveImg = cv2.resize(interleaveImg, (imW, imH))
            image[i] = interleaveImg[i]
            image[:, j] = interleaveImg[:, j]
        return dict(image=image, **kw)

    return __prcs


def ChangeOrBlurBackground(bgValue=0, minAlpha=0.75, p=0.5):
    '''
    bgValue: pixel value which replace background, those pixels will be replace with bgImg
    minAlpha: (0.75 to 1.0); will crease superimposition of two image and bgImg
            mixAlpha = np.random.uniform(minAlpha, 1)
            image = cv2.addWeighted(image, mixAlpha, bgImg, 1 - mixAlpha, 0)
    bgImg: image whose pixels will be used to replace the background, if the value is None it will blur current background
    '''
    bgValue = np.array(bgValue)

    def __prcs(image, mask=None, masks=None, bboxes=None, bgImg=None, force_apply=False, **kw):
        if force_apply or random.random() < p:
            if bgImg is None:
                bgImg = cv2.blur(image, (int(random.uniform(11, 21)), int(random.uniform(11, 21))))
            else:
                imH, imW = image.shape[:2]
                bgImg = cv2.resize(bgImg, (imW, imH))
            if masks is not None:
                image = __cutMask(image, masks, bgImg, isMasks=True)
            elif mask is not None:
                image = __cutMask(image, mask, bgImg, isMasks=False)
            elif bboxes:
                image = __cutBbox(image, bboxes, bgImg)
            else:
                raise Exception("""
                Either mask or masks or bboxes should be passed
                """)
        return dict(image=image, mask=mask, masks=masks, bboxes=bboxes, bgImg=bgImg, **kw)

    def __cutMask(image, mask, bgImg, isMasks):
        if isMasks:
            roi = mask[0] > 0
            for m in mask[1:]:
                roi |= m > 0
            roi = ~roi
        else:
            if len(mask.shape) == 3:
                roi = np.all(mask == bgValue, axis=-1)
            else:
                roi = mask == bgValue
        mixAlpha = np.random.uniform(minAlpha, 1)
        image = cv2.addWeighted(image, mixAlpha, bgImg, 1 - mixAlpha, 0)
        image[roi] = bgImg[roi]
        return image

    def __cutBbox(image, bboxes, bgImg):
        imH, imW = image.shape[:2]
        for x0, y0, x1, y1, label in bboxes:
            # assert 0 <= x0 < 1 and 0 <= y0 < 1 and 0 < x1 <= 1 and 0 < y1 <= 1
            x0, y0, x1, y1 = int(x0 * imW), int(y0 * imH), int(x1 * imW), int(y1 * imH)
            mixAlpha = np.random.uniform(minAlpha, 1)
            bgImg[y0:y1, x0:x1] = cv2.addWeighted(image[y0:y1, x0:x1], mixAlpha, bgImg[y0:y1, x0:x1], 1 - mixAlpha, 0)
        image = bgImg
        return image

    return __prcs


def ImageCollage(collageT):
    '''
    collageT: A A.Compose object. Before applying collage, every sub-image will undergo this transform
    '''
    r2c2 = {0: [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]],
            1: [[0.5, 0.5], [0.5, 0.0], [0.5, 0.5], [0.5, 0.0]],
            2: [[0.5, 0.0], [0.5, 0.5], [0.5, 0.0], [0.5, 0.5]],
            3: [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]}
    r2c1 = {0: [[1.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.0]],
            1: [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.5, 0.5]]}
    r1c2 = {0: [[0.5, 0.0], [1.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            1: [[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]}

    def __prcs(datas):
        if len(datas) == 4:
            shiftBy, nRow, nCol = r2c2, 2, 2
        else:
            shiftBy, nRow, nCol = random.choice([[r2c1, 2, 1], [r1c2, 1, 2]])
        for d in datas:
            if d.get('masks'):
                raise Exception("""
                masks is not supported
                """)
        datas = [collageT(**data) for data in datas]
        image = __mergeData(datas, 'image', nCol)
        mask = __mergeData(datas, 'mask', nCol)
        collageData = dict(image=image, mask=mask, bboxes=[])
        collageData = __collageBox(datas, collageData, shiftBy)
        return collageData

    def __collageBox(datas, collageData, shiftBy):
        for quater, data in enumerate(datas):
            bboxes = data.get('bboxes', [])
            try:
                for x0, y0, x1, y1, label in bboxes:
                    # assert 0 <= x0 < 1 and 0 <= y0 < 1 and 0 < x1 <= 1 and 0 < y1 <= 1
                    x0, y0, x1, y1 = [m * x + c for x, (m, c) in zip([x0, y0, x1, y1], shiftBy[quater])]
                    collageData['bboxes'].append([x0, y0, x1, y1, label])
            except:
                raise Exception(f"""
                        expected albumentations format [x_min, y_min, x_max, y_max, label]
                        all values should be in the range of 0 to  1
                        """)
        return collageData

    def __mergeData(datas, key, nCol):
        imCol, imRow = [], []
        for ix, d in enumerate(datas, -1):
            i = d.get(key)
            if i is not None:
                imCol.append(i)
                if ix % nCol == 0:
                    imRow.append(cv2.hconcat(imCol))
                    imCol = []
        img = cv2.vconcat(imRow)
        return img

    return __prcs


def image_copy_paste(img, paste_img, alpha, alphaMix, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
        img_dtype = img.dtype
        alpha = alpha[..., None] * alphaMix
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)
    return img


def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]
        masks.extend(paste_masks)
    return masks


def extract_bboxes(masks):
    bboxes = []
    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        x0y0x1y1 = 0, 0, 0, 0
        if yindices.shape[0]:
            x0, x1 = yindices[[0, -1]]
            y0, y1 = xindices[[0, -1]]
            x0y0x1y1 = x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h
        bboxes.append(x0y0x1y1)
    return bboxes


def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha):
    if paste_bboxes is not None:
        masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)
        adjusted_bboxes = extract_bboxes(masks)
        # only keep the bounding boxes for objects listed in bboxes
        mask_indices = [box[-1] for box in bboxes]
        adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        # append bbox tails (classes, etc.)
        adjusted_bboxes = [bbox + tail[4:] for bbox, tail in zip(adjusted_bboxes, bboxes)]
        # adjust paste_bboxes mask indices to avoid overlap
        max_mask_index = len(masks)
        paste_mask_indices = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        paste_bboxes = [pbox[:-1] + (pmi,) for pbox, pmi in zip(paste_bboxes, paste_mask_indices)]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        adjusted_paste_bboxes = [apbox + tail[4:] for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)]
        bboxes = adjusted_bboxes + adjusted_paste_bboxes
    return bboxes


def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    # remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)
        keypoints = visible_keypoints + paste_keypoints
    return keypoints


def CopyPaste(blend=True, sigma=3, pct_objects_paste=1.0, max_paste_objects=np.inf, minAlpha=.75, p=0.5):
    '''
    blend = True and sigma: (0.0 to 5.0) the border of objects will be blur to have smooth paste effect
    pct_objects_paste: (0.0 to 1.0) if pct_objects_paste = 0.5 half of objects will randomly choose for copying
    max_paste_objects: (0 to inf) if max_paste_objects = 3 at max 3 objects will be randomly pasted on src image
    minAlpha: (0.75 to 1.0); will crease superimposition of two image and bgImg
        mixAlpha = np.random.uniform(minAlpha, 1)
        image = cv2.addWeighted(image, mixAlpha, bgImg, 1 - mixAlpha, 0)
    paste_image = image used to paste
    paste_masks = its corresponding masks (mask/paste_mask is not supported please split as masks it before using copy paste)
    paste_bboxes (optional) bounding boxes
    '''
    def __prcs(image, masks, mask=None, bboxes=None, paste_image=None, paste_mask=None, paste_masks=None, paste_bboxes=None, force_apply=False, **kw):
        if mask is not None or paste_mask is not None:
            raise Exception(f"""
            mask is not supported please split it and pass as masks argument
            """)
        if force_apply or random.random() < p:
            if paste_bboxes and not paste_masks:
                raise Exception("""
                For proper working of paste_bboxes paste_masks are required.
                Please try other augmentation like ReplaceBackground""")
            ix = range(len(paste_bboxes)) if paste_bboxes else range(len(paste_masks))
            n_objects = len(ix)
            n_select = min(int(n_objects * pct_objects_paste), max_paste_objects)
            ix = np.random.choice(ix, size=n_select, replace=False)
            if paste_bboxes:
                paste_bboxes = [paste_bboxes[i] for i in ix]
                ix = [bbox[-1] for bbox in paste_bboxes]
            alpha = None
            if paste_masks:
                paste_masks = [paste_masks[i] for i in ix]
                alpha = paste_masks[0] > 0
                for mask in paste_masks[1:]:
                    alpha += mask > 0

            bboxes = bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha)
            image = image_copy_paste(image, paste_image, alpha, random.uniform(minAlpha, 1), blend=blend, sigma=sigma)
            masks = masks_copy_paste(masks, paste_masks, alpha)
        return dict(image=image, masks=masks, bboxes=bboxes, paste_image=paste_image, paste_masks=paste_masks, paste_bboxes=paste_bboxes, **kw)

    return __prcs


def augTest(data, bgImg, interleaveImg):
    transformT = A.Compose([
        InterleaveAug(p=1),
        ChangeOrBlurBackground(bgValue=0, minAlpha=.75, p=1),
        CopyPaste(p=1),
    ], bbox_params=A.BboxParams('albumentations', min_visibility=.05))
    t = transformT(interleaveImg=interleaveImg, bgImg=bgImg, **data)
    return t


def collageTest(datas):
    bgColor = [0, 0, 0]
    subImgWidth, subImgHeight = 384, 384
    collageT = A.Compose([
        A.Resize(height=subImgHeight, width=subImgWidth, p=1),
        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=bgColor, p=.5, scale_limit=[-.4, -.2], shift_limit=[-.4, -.2], rotate_limit=15),
        A.HorizontalFlip(),
    ], bbox_params=A.BboxParams('albumentations', min_visibility=.05))

    collageIt = ImageCollage(collageT)
    t = collageIt(datas)
    return t
