import os
import re
import ast
import time
import json
import math
import pickle
import hashlib
import argparse
from functools import partial
from itertools import groupby
from itertools import zip_longest
from itertools import permutations
from itertools import combinations
from .pixCommon import *
from .bashIt import *

try:
    import yaml
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass

try:
    import dlib
except:
    pass

try:
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda xFormatXpw13: f'{xFormatXpw13:8.4f}')
except:
    pass


class tqdm:
    """
    fix: notbook extra line issue
    """
    from tqdm import tqdm

    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=sys.stdout, ncols=None, mininterval=0.1, maxinterval=10.0,
                 miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3,
                 bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=None, lock_args=None, gui=False, **kwargs):
        self.args = dict(
                iterable=iterable, desc=desc, total=total, leave=leave, file=file, ncols=ncols, mininterval=mininterval, maxinterval=maxinterval,
                miniters=miniters, ascii=ascii, disable=disable, unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                bar_format=bar_format, initial=initial, position=position, postfix=postfix, unit_divisor=unit_divisor, write_bytes=write_bytes, lock_args=lock_args, gui=gui)
        self.args.update(kwargs)
        self.pbar = None

    def __iter__(self):
        iterable = self.args['iterable']
        with self.tqdm(**self.args) as self.pbar:
            for i in iterable:
                yield i
                self.pbar.update(1)

    def set_msg(self, description=None, postFix=None):
        if description is not None:
            self.pbar.set_description(description)
        if postFix is not None:
            self.pbar.set_postfix(postFix)


class DotDict(dict):
    def __init__(self, datas=None):
        super().__init__()
        if isinstance(datas, argparse.Namespace):
            datas = vars(datas)
        datas = dict() if datas is None else datas
        if datas:
            self.update(DotDict.dict2DotDict(datas))

    @staticmethod
    def dict2DotDict(datas):
        dotDict = DotDict()
        for k, v in datas.items():
            if isinstance(v, dict):
                dotDict[k] = DotDict.dict2DotDict(v)
            else:
                dotDict[k] = v
        return dotDict

    def __getattr__(self, key):
        if key not in self:
            print("56 __getattr__ pixCommon key: ", key)
            raise AttributeError(key)
        else:
            return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        dictData = self
        data = self.dict2str(dictData)
        return data

    def dict2str(self, dictData, prefix=''):
        keys = list(dictData.keys())
        nSpace = len(max(keys, key=lambda x: len(x))) + 2
        keys = sorted(keys)
        data = []
        for key in keys:
            val = dictData[key]
            if isinstance(val, dict):
                val = self.dict2str(val, prefix=f"\t")
            elif type(val) == str:
                val = f"'{val}'"
            key = f"{prefix}{key}"
            data.append(f'{key:{nSpace}}: {val}')
        data = '\n%s\n' % '\n'.join(data)
        return data

    def copy(self):
        return DotDict(super().copy())

    def toJson(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return json.dumps(res)

    def toDict(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return res


def readYaml(src, defaultDict=FileNotFoundError, returnDotDict=True):
    src = getPath(src)
    if defaultDict == FileNotFoundError:
        assert exists(src)
    with open(src, 'r') as book:
        data = yaml.safe_load(book)
    if returnDotDict:
        data = DotDict(data)
    return data


def writeYaml(yamlPath, jObjs):
    with open(dirop(yamlPath), 'w') as book:
        yaml.dump(jObjs, book, default_flow_style=False, sort_keys=False)


def readBook(bookPath, returnList=True, mode='r', encoding=None, errors=None):
    with open(getPath(bookPath), mode=mode, encoding=encoding, errors=errors) as f:
        lines = f.read()
    lines = lines.strip()
    if returnList:
        lines = lines.split('\n')
    return lines


def writeBook(bookPath, msg, mode='w', encoding=None, errors=None):
    with open(dirop(bookPath), mode=mode, encoding=encoding, errors=errors) as book:
        book.write(msg)


def readPkl(pklPath, defaultData=None, writePkl=True, rm=False, verbose=True):
    pklPath = getPath(pklPath)
    if rm and os.path.exists(pklPath):
        print(f"deleting pklPath: {pklPath}")
        os.remove(pklPath)
    pklExists = os.path.exists(pklPath)
    if pklExists:
        if verbose:
            print(f"loading pklPath: {pklPath}")
        defaultData = pickle.load(open(pklPath, 'rb'))
    elif callable(defaultData):
        if verbose:
            print(f"executing: {defaultData}")
        defaultData = defaultData()
    if defaultData is not None and writePkl and not pklExists:
        if verbose:
            print(f"pkling defaultData: {pklPath}")
        pickle.dump(defaultData, open(dirop(pklPath), 'wb'))
    return defaultData


def writePkl(pklPath, objs):
    pickle.dump(objs, open(dirop(pklPath), 'wb'))


def p2l(x, idx=None, noExt=False):
    if noExt:
        x = os.path.splitext(x)[0]
    p = [p for p in x.split(os.sep)]
    if idx is not None:
        p = [p[i] for i in idx]
    return p


def l2p(x, ext='', joinBy=os.sep):
    assert type(x) != str, f"input cannot be string"
    assert not ext.startswith('.'), f"ext is starting with dot {ext}"
    ext = ''.join(ext)
    x = joinBy.join(x)
    if ext:
        x = f"{x}.{ext}"
    return x


def dir2(var):
    """
    list all the methods and attributes present in object
    """
    for v in dir(var):
        print(v)
    print("34 dir2 common : ", )
    quit()


def float2img(img, min=None, max=None):
    min = img.min() if min is None else min
    max = img.max() if max is None else max
    img = img.astype('f4')
    img -= min
    img /= max
    return (255 * img).astype('u1')


def bboxLabel(img, txt="", loc=(30, 45), color=(255, 255, 255), thickness=3, txtSize=1, txtFont=None, txtThickness=1, txtColor=None, asTitle=None):
    txtFont = cv2.QT_FONT_NORMAL if txtFont is None else txtFont
    if len(loc) == 4:
        x0, y0, w, h = loc
        x0, y0, rw, rh = int(x0), int(y0), int(w), int(h)
        cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), list(color), thickness)
    else:
        if asTitle is None:
            asTitle = True
        x0, y0, rw, rh = int(loc[0]), int(loc[1]), 0, 0
    txt = str(txt)
    if txt != "":
        if txtColor is None:
            txtColor = (0, 0, 0)
        (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
        # baseLine -> to fit char like p,y in box
        if asTitle:
            h, w = img.shape[:2]
            zimg = np.zeros([60, w, 3], 'u1') + 255
            cv2.putText(zimg, txt, (x0, y0 + rh - baseLine), txtFont, txtSize, txtColor, txtThickness, cv2.LINE_AA)
            img = cv2.vconcat([zimg, img])
        else:
            cv2.rectangle(img, (x0, y0 + rh), (x0 + w, y0 + rh - h - baseLine), color, -1)
            cv2.putText(img, txt, (x0, y0 + rh - baseLine), txtFont, txtSize, txtColor, txtThickness, cv2.LINE_AA)
    return img


def bboxScale(img, bbox, scaleWH):
    try:
        sw, sh = scaleWH
    except:
        sw, sh = scaleWH, scaleWH
    x, y, w, h = bbox
    xc, yc = (x + w / 2, y + h / 2)
    w *= sw
    h *= sh
    x, y = xc - w / 2, yc - h / 2
    return frameFit(img, (x, y, w, h))


def frameFit(img, bbox):
    """
    ensure the bbox will not go away from the image boundary
    """
    imHeight, imWidht = img.shape[:2]
    x0, y0, width, height = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = x0 + int(width), y0 + int(height)
    x1, y1 = min(x1, imWidht), min(y1, imHeight)
    return np.array((x0, y0, max(0, x1 - x0), max(0, y1 - y0)))


def putSubImg(mainImg, subImg, loc, interpolation=None):
    """
    place the sub image inside the genFrame image
    """
    interpolation = cv2.INTER_CUBIC if interpolation is None else interpolation
    if len(loc) == 2:
        x, y = int(loc[0]), int(loc[1])
        h, w = subImg.shape[:2]
    else:
        x, y, w, h = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
        subImg = cv2.resize(subImg, (w, h), interpolation=interpolation)
    x, y, w, h = frameFit(mainImg, (x, y, w, h))
    mainImg[y:y + h, x:x + w] = getSubImg(subImg, (0, 0, w, h))
    return mainImg


def getSubImg(im1, bbox):
    """
    crop sub image from the given input image and bbox
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = im1[y:y + h, x:x + w]
    if img.shape[0] and img.shape[1]:
        return img


class VideoWriter:
    """mjpg xvid mp4v"""

    def __init__(self, path, camFps, size=None, codec='mp4v'):
        self.path = path
        try:
            self.fps = camFps.get(cv2.CAP_PROP_FPS)
        except:
            self.fps = camFps
        self.__vWriter = None
        self.__size = size
        self.__codec = cv2.VideoWriter_fourcc(*(codec.upper()))
        print("writing :", path, '@', self.fps, 'fps')

    def write(self, img):
        if self.__vWriter is None:
            if self.__size is None:
                self.__size = tuple(img.shape[:2])
            self.__vWriter = cv2.VideoWriter(dirop(self.path), self.__codec, self.fps, self.__size[::-1])
        if tuple(img.shape[:2]) != self.__size:
            img = cv2.resize(img, self.__size[::-1])
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.__vWriter.write(img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.__vWriter:
            self.__vWriter.release()
        else:
            print(f"video: {self.path} closed without opening")

    def __enter__(self):
        return self


class Wait:
    def __init__(self):
        self.pause = False

    def __call__(self, delay=1):
        if self.pause:
            delay = 0
        key = cv2.waitKey(delay)
        if key == 32:
            self.pause = True
        if key == 13:
            self.pause = False
        return key


__wait = Wait()


def showImg(winname='output', imC=None, delay=None, windowConfig=0, nRow=None, chFirst=False, useMatplot=False):
    winname = str(winname)
    if imC is not None:
        if type(imC) is not list:
            imC = [imC]
        imC = photoframe(imC, nRow=nRow, chFirst=chFirst)
        if not useMatplot:
            try:
                cv2.namedWindow(winname, windowConfig)
                cv2.imshow(winname, imC)
            except Exception as exp:
                if 'The function is not implemented' in str(exp):
                    print(exp)
                    print("forcing to use matplotlib")
                    useMatplot = True
                else:
                    raise Exception(exp)
        if useMatplot:
            plt.imshow(imC)
            plt.show()
        # else:

    if not useMatplot and delay is not None:
        key = __wait(delay)
        return key
    return imC


def photoframe(imgs, rcsize=None, nRow=None, resize_method=None, fit=False, asgray=False, chFirst=False):
    """
    # This method pack the array of images in a visually pleasing manner.
    # If the nCol is not specified then the nRow and nCol are equally divided
    # This method can automatically pack images of different size. Default stitch size is 128,128
    # when fit is True final photo frame size will be rcsize
    #          is False individual image size will be rcsize
    # Examples
    # --------
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=None)
        for fnos, imgs in video.chunk(4):
            i1 = photoframe(imgs, nCol=None)
            i2 = photoframe(imgs, nCol=4)
            i3 = photoframe(imgs, nCol=4, rcsize=(200,300),nimgs=7)
            i4 = photoframe(imgs, nCol=3, nimgs=7)
            i5 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape)
            i6 = photoframe(imgs, nCol=6, rcsize=imgs[0].shape, fit=True)
            i7 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape, fit=True, asgray=True)
            for i, oldFeature in enumerate([i1, i2, i3, i4, i5, i6, i7], 1):
                print(i, oldFeature.shape)
                win('i%s' % i, )(oldFeature)
            win('totoal')(photoframe([i1, i2, i3, i4, i5, i6, i7]))
            if win().__wait(waittime) == 'esc':
                break
    """
    resize_method = cv2.INTER_LINEAR if resize_method is None else resize_method
    if len(imgs):
        if chFirst:
            imgs = np.array([np.transpose(img, [1, 2, 0]) for img in imgs])
        if rcsize is None:
            rcsize = imgs[0].shape
        imrow, imcol = rcsize[:2]  # fetch first two vals
        nimgs = len(imgs)
        nRow = int(np.ceil(nimgs ** .5)) if nRow is None else max(1, int(nRow))
        nCol = nimgs / nRow
        nCol = int(np.ceil(nCol + 1)) if (nRow * nCol) - nimgs else int(np.ceil(nCol))
        if fit:
            imrow /= nRow
            imcol /= nCol
        imrow, imcol = int(imrow), int(imcol)
        resshape = (imrow, imcol) if asgray else (imrow, imcol, 3)
        imgs = zip_longest(list(range(nRow * nCol)), imgs, fillvalue=np.zeros(resshape, imgs[0].dtype))
        resimg = []
        for _, imggroup in groupby(imgs, lambda k: k[0] // nCol):
            rowimg = []
            for _, img in imggroup:
                if img.dtype != np.uint8:
                    print("warning float2img may not work properly")
                    img = float2img(img)
                if asgray:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 1:
                    img = img.reshape(*img.shape[:2])
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if tuple(img.shape) != resshape:
                    img = cv2.resize(img, (imcol, imrow), interpolation=resize_method)
                rowimg.append(img)
            resimg.append(cv2.hconcat(rowimg))
        return cv2.vconcat(resimg)


def prr(name, img, getVal=False):
    torch, tf = None, None
    try:
        import torch  # TODO remove this or add tf support
    except Exception as exp:
        pass
    try:
        import tensorflow as tf
    except:
        pass

    if type(img) == list:
        if torch and isinstance(img[0], torch.Tensor):
            img = torch.stack(img)
        elif isinstance(img[0], np.ndarray):
            img = np.array(img)
        elif tf and isinstance(img[0], tf.Tensor):
            img = tf.stack(img)
        else:
            raise Exception(f"data type of {type(img[0])} not implemented")
    dtype = str(img.dtype)
    try:
        dtype = f"{dtype}{img.device[-5:]}"
    except:
        pass
    unique = []
    if getVal:
        try:
            unique = torch.unique(img.detach())
        except:
            unique = np.unique(img)
    nUnique = len(unique)
    step = max(1, nUnique // 20)
    nEle = 1
    for i in img.shape:
        nEle *= i
    shape = f"{nEle}={str(list(img.shape))}"
    try:
        data = f"{name:45s}:{dtype:15s} [{img.min():6.3f} {img.max():6.3f}] {shape:30s} [{nUnique} {unique[::step]}]"
    except:
        data = f"{name:45s}:{dtype:15s} [{np.min(img):6.3f} {np.max(img):6.3f}] {shape:30s} [{nUnique} {unique[::step]}]"
    print(data)


def dispVars(names, kwargs):
    # print(f"_____________________{names}_____________________")
    # for k, v in kwargs.items():
    #     try:
    #         print(f"{k}={v},", end=end)
    #     except Exception as exp:
    #         pass

    x = pd.DataFrame(kwargs.items(), columns=['variable', 'val'])
    nDelim = 0
    res = []
    for v in x.val:
        v = str(v)
        n = len(v)
        if nDelim < n:
            nDelim = n
        res.append([n, v])
    x['val'] = [f"{v}{' ' * (nDelim - n - 1)}" for n, v in res]
    x.index = x['variable']
    x.index.name = names
    print(x[['val']])


class clk:
    def __init__(self, roundBy=4):
        self.__roundBy = roundBy
        self.__toks = [["start", dt.now()]]

    def tok(self, name=None):
        if name is None:
            n = len(self.__toks)
            name = f'{n}_{n + 1}'
        self.__toks.append([name, dt.now()])
        return self

    def __repr__(self):
        cols, datas = self.get()
        try:
            datas = pd.DataFrame(datas, columns=cols).round(self.__roundBy)
        except:
            print(cols)
            datas = np.array(datas)
            datas[:, 1:] = datas[:, 1:].astype(float).round(self.__roundBy)
            datas = datas.astype(str)
        return str(datas)

    def get(self, returnType='pd'):
        toks = self.__toks
        datas = []
        cols = 'name', 'fps', 'sec'
        for (pName, tik), (name, tok) in zip(toks, toks[1:]):
            lap = self.__getLap(tik, tok)
            data = f"{pName}_{name}", 1 / lap, lap
            datas.append(data)
        if returnType == 'np':
            datas = cols, np.array(datas)
        elif returnType == 'pd':
            datas = pd.DataFrame(datas, columns=cols)
            datas = pd.DataFrame(datas, columns=cols)
            datas = datas.set_index('name')
            lapse = self.__getLap(toks[0][-1], toks[-1][-1])
            datas.index.name = f"{lapse :0.3f}[{1/lapse :0.3f}]"
        elif returnType == 'list':
            datas = cols, datas
        return datas

    def last(self, roundBy=None):
        roundBy = roundBy or self.__roundBy
        toks = self.__toks
        (_, tik), (_, tok) = toks[-2], toks[-1]
        lap = self.__getLap(tik, tok)
        return round(lap, roundBy)

    @staticmethod
    def __getLap(tik, tok):
        lap = tok - tik
        lap = lap.seconds + (lap.microseconds / 1000000)
        return lap


def zipIt(src, desZip, rm=False):
    src, desZip = getPath(src), getPath(desZip)
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if exists(desZip):
        if rm:
            os.remove(desZip)
        else:
            raise Exception(f'''Fail des: {desZip} \n\talready exists delete it before operation''')
    desZip, zipExt = os.path.splitext(desZip)
    if os.path.isfile(src):
        tempDir = join(dirname(src), getTimeStamp())
        if os.path.exists(tempDir):
            raise Exception(f'''Fail tempDir: {tempDir} \n\talready exists delete it before operation''')
        os.makedirs(tempDir)
        shutil.copy(src, tempDir)
        desZip = shutil.make_archive(desZip, zipExt[1:], tempDir)
        shutil.rmtree(tempDir, ignore_errors=True)
    else:
        desZip = shutil.make_archive(desZip, zipExt[1:], src)
    return desZip


def unzipIt(src, desDir=None, rm=False):
    desDir = desDir or dirname(src)
    src, desDir = getPath(src), getPath(desDir)
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if os.path.splitext(desDir)[-1]:
        raise Exception(f'''Fail desDir: {desDir} \n\tshould be folder''')
    tempDir = f"{desDir}/{getTimeStamp()}"
    shutil.unpack_archive(src, tempDir)
    if not exists(desDir):
        os.makedirs(desDir)
    for mvSrc in os.listdir(tempDir):
        mvSrc = join(tempDir, mvSrc)
        mvDes = join(desDir, basename(mvSrc))
        if rm and exists(mvDes):
            if os.path.isfile(mvDes):
                os.remove(mvDes)
            else:
                shutil.rmtree(mvDes, ignore_errors=True)
        try:
            shutil.move(str(mvSrc), desDir)
        except Exception as exp:
            raise Exception(f"""
{exp}
tempDir: {tempDir}
""")
    shutil.rmtree(tempDir, ignore_errors=True)
    return desDir


def getGdownCmd(downloadCmd, url):
    # pip install bs4 lxml gdown
    if 'Permission denied' in url:
        import requests
        try:
            from bs4 import BeautifulSoup
        except Exception as exp:
            raise Exception(f"""{exp}

            try: pip install bs4 lxml gdown
            """)
        for dirId in url.split('Permission denied: '):
            dirId = dirId.split(' Maybe you need')[0]
            if 'https' in dirId:
                dirId = dirId.split('id=')[1]
                r = requests.get(f'https://drive.google.com/drive/folders/{dirId}?usp=sharing')
                for glink in str(BeautifulSoup(r.text, "lxml")).split('data-id="'):
                    if 'data-target=' in glink:
                        gid = glink.split('" data-target')[0]
                        downloadCmd += f'gdown https://drive.google.com/uc?id={gid};'
        for i in downloadCmd.split(';'):
            print(i)
    elif 'folder' in url:
        import requests
        try:
            from bs4 import BeautifulSoup
        except Exception as exp:
            raise Exception(f"""{exp}

                try: pip install bs4 lxml gdown
                """)
        r = requests.get(url)
        for glink in str(BeautifulSoup(r.text, "lxml")).split('data-id="'):
            if 'data-target=' in glink:
                gid = glink.split('" data-target')[0]
                downloadCmd += f'gdown https://drive.google.com/uc?id={gid};'
    else:
        if '/d/' in url:
            gid = url
            gid = gid.split('/d/')[1]
            gid = gid.split('/')[0]
        elif 'id=' in url:
            gid = url
            gid = gid.split('id=')[1]
            gid = gid.split('&')[0]
        downloadCmd += f'gdown https://drive.google.com/uc?id={gid};'
    return downloadCmd


def rawsGlob(s3path, raiseOnEmpty=True):
    assert '*' in s3path, f"* missing {s3path}"
    root, path = [], []
    for x in s3path.split(os.sep):
        if not path and '*' not in x:
            root.append(x)
        else:
            path.append(x)
    cmd, errCode, out, err = exeIt(f'aws s3 sync {os.sep.join(root)} . --exclude "*" --include "{os.sep.join(path)}" --dryrun', debug=False)
    if raiseOnEmpty:
        assert out, f"file not found: {s3path}"
    return [x.split('download: ')[1].split(' to ')[0] for x in out.split('\n')] if out else []


class AccessS3:

    @staticmethod
    def s3setup(isDir, local, localRoot, remoteRoot, skipExe, rm):
        awsKey = f"export AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']};export AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"
        local = getPath(local).rstrip('/')
        localRoot, remoteRoot = getPath(localRoot or dirname(local)).rstrip('/'), remoteRoot.rstrip('/')
        remoteRoot = remoteRoot if remoteRoot != 's3:' else 's3:/'
        remote = f"{remoteRoot}{local[len(localRoot):]}"
        fType = 'folder' if isDir else 'file'
        assert remoteRoot.startswith('s3://')
        assert local.startswith(localRoot), f"path, {local} should start with {localRoot}"
        if exists(getPath('~/.ssh/local')):
            if input("Do you want to upload/download from your local machine? [y/n]:\t") != 'y':
                skipExe = True
        if rm:
            if input("Do you want to delete file before upload/download? [y/n]:\t") != 'y':
                skipExe = True
        return skipExe, awsKey, fType, local, remote

    @staticmethod
    def up(isDir, local, remoteRoot, localRoot=None, *, desName=None, rmRemote=False, skipExe=False, debug=False):
        skipExe, awsKey, fType, local, remote = AccessS3.s3setup(isDir, local, localRoot, remoteRoot, skipExe, rmRemote)
        remote = f"{dirname(remote)}/{basename(desName or local)}"
        assert os.path.exists(local), f"\npath not exists: {local}"
        assert os.path.isdir(local) == bool(isDir), f"\n{local} is {'folder' if os.path.isdir(local) else 'file'} set \n isDir = {os.path.isdir(local)}"

        awsCmd, awsRmCmd = f'aws s3 {"sync" if isDir else "cp"} {local} {remote}', f'aws s3 rm {remote} {"--recursive" if isDir else ""}'
        print(f"uploading[{fType}] : {awsCmd}")
        if not skipExe:
            if rmRemote:
                AccessS3.s3cmdExe(awsKey, awsRmCmd, debug)
            AccessS3.s3cmdExe(awsKey, awsCmd, debug)
        return remote

    @staticmethod
    def down(isDir, local, remoteRoot, localRoot=None, *, desName=None, rmLocal=False, skipExe=False, debug=False):
        skipExe, awsKey, fType, local, remote = AccessS3.s3setup(isDir, local, localRoot, remoteRoot, skipExe, rmLocal)
        local = f"{dirname(local)}/{basename(desName or remote)}"
        awsCmd = f'aws s3 {"sync" if isDir else "cp"} {remote} {local}'

        print(f"downloading[{fType}] : {awsCmd}")
        if not skipExe:
            if rmLocal:
                dirop(local, rm=True)
            AccessS3.s3cmdExe(awsKey, awsCmd, debug)
            if not isDir:
                try:
                    local = unzipIt(local)
                except:
                    print(f"failed to unzip: {local}")
        return local

    @staticmethod
    def s3cmdExe(awsKey, awsCmd, debug):
        if debug:
            exeIt(f'{awsKey};{awsCmd}', debug=True, returnOutput=True)
        else:
            exeIt(f'{awsKey};{awsCmd} > /dev/null 2>&1', debug=False, returnOutput=False)


def randomIt(x, dtype, asTorch=False, xmin=-100, xmax=100):
    try:
        x = np.random.randint(xmin, xmax, x, dtype)
    except:
        x = np.random.randint(xmin * 1000, xmax * 1000, x, 'i4').astype(dtype) / 1000
    if asTorch:
        import torch
        x = torch.from_numpy(x)
    return x


def downloadDB(url, desDir, cache=None, unzip=True, skipExe=False):
    url = url.strip()
    if cache and exists(getPath(cache)):
        returnData = [getPath(cache)]
    else:
        desDir = getPath(desDir)
        dirop(desDir)
        done, downloadCmd = False, f'cd "{desDir}";'
        old = set(glob(f'{desDir}/*'))
        if url.startswith('git+'):
            downloadCmd += f'git clone "{url.lstrip("git+").lstrip()}";'
        elif url.startswith('gdrive+'):
            url = url.lstrip('gdrive+').lstrip()
            downloadCmd = getGdownCmd(downloadCmd, url)
        elif url.startswith('gdown+'):
            url = url.lstrip('gdown+').lstrip()
            downloadCmd = getGdownCmd(downloadCmd, url)
        elif url.startswith('youtube+'):
            downloadCmd += f"youtube-dl '{url.lstrip('youtube+').lstrip()}' --print-json --restrict-filenames -o '%(id)s.%(ext)s'"
        elif url.startswith('wgetNoCertificate+'):
            downloadCmd += f'wget --no-check-certificate "{url.lstrip("wgetNoCertificate+").lstrip()}";'
        elif url.startswith('wget+'):
            downloadCmd += f'wget "{url.lstrip("wget+").lstrip()}";'
        else:
            raise Exception(f"unknown url format: {url}")
        if not skipExe:
            print("____________________________________________________________________________________________________________________")
            print(f"\n             {downloadCmd}\n")
            print("____________________________________________________________________________________________________________________")
        if not done:
            exeIt(downloadCmd, returnOutput=False, debug=False, skipExe=skipExe)
        returnData = list(set(glob(f'{desDir}/*')) - old)
    if len(returnData) != 1:
        print("skipping unzip no. file downloaded != 1")
        print("returnData:", returnData)
        return returnData
    returnData = returnData[0]
    if unzip:
        try:
            print("unzip dataset")
            print("zip: ", returnData)
            returnData = unzipIt(returnData)
        except Exception as exp:
            print(f"skipping unzip: {returnData}")
            print(f"""
Exception:
        {exp}
                """)
    print("returnData: ", returnData)
    return returnData


def replaces(path, *words):
    path = str(path)
    assert len(words) % 2 == 0
    words = zip(words[::2], words[1::2])
    for word in words:
        path = path.replace(*word)
    return path


def compareVersions(versions, compareBy, ovideoPlayer=None, putTitle=bboxLabel, bbox=None, showDiff=False):
    vPlayer = videoPlayer if ovideoPlayer is None else ovideoPlayer
    vpaths = [compareBy] + [version for version in versions if version != compareBy]
    vplayers = [vPlayer(cv2.VideoCapture(version)) for version in vpaths]
    for _, data in enumerate(zip(*vplayers)):
        imgs = []
        for vpath, (fno, ftm, img) in zip(vpaths, data):
            img = imResize(img, (780, 1280))
            if bbox is True:
                winname = "select roi"
                cv2.namedWindow(winname, 0)
                bbox = cv2.selectROI(winname, img)
                cv2.destroyWindow(winname)
            if bbox is not None:
                img = getSubImg(img, bbox)
            # img = bboxLabel(img, basename(vpath))
            imgs.append(img)
        datas = []
        for ix, img in enumerate(imgs[1:], 1):
            res = []
            score = 0
            res.append(putTitle(imgs[0].copy(), basename(vpaths[0])))
            res.append(putTitle(imgs[ix].copy(), basename(vpaths[ix])))
            imgs[0] = putSubImg(imgs[0], np.zeros_like(imgs[0]), (0, 100, 200, 200))
            imgs[0] = putSubImg(imgs[0], np.zeros_like(imgs[0]), (0, 0, 1200, 100))
            if showDiff:
                img = putSubImg(img, np.zeros_like(img), (0, 100, 200, 200))
                img = putSubImg(img, np.zeros_like(img), (0, 0, 1200, 100))
                diff = cv2.absdiff(imgs[0], img)
                res.append(diff)
                diff = cv2.inRange(diff.min(axis=-1), 10, 300)
                score = diff.mean()
                res.append(diff)
            datas.append([score, res])
        yield datas


def getGitPushPull():
    def gitClone(rootDir, cloneCmd, skipExe=False):
        sepBy = '; '
        dirop(rootDir)
        cmd = f'''
    cd {dirname(rootDir)}
    {cloneCmd}
        '''
        exeIt(cmd, sepBy=sepBy, skipExe=skipExe)

    def gitPull(rootDir, commitId, skipExe=False, pullLocal=False):
        if not pullLocal and not os.getcwd().startswith('/home/ec2-user'):
            raise Exception("pull works only on ec2-user")
        sepBy = '; '
        dirop(rootDir)
        if commitId.startswith('git clone'):
            gitClone(rootDir, commitId)
        else:
            cmd = f"""
cd {rootDir}
git reset --hard {commitId}
        """
            exeIt(cmd, returnOutput=True, sepBy=sepBy, debug=True, skipExe=skipExe)
        print("15 gitPull main : ", )
        quit()

    def gitPush(rootDir, message, gitAdds=None, cloneCmd=None, removeCache=True, stop=False, skipExe=False):
        if stop:
            print(f"cloneCmd   :\t rm -rf /tmp/gitSync;mkdir -p /tmp/gitSync;cd /tmp/gitSync;git clone git@bitbucket.org:vishnuChand/aws.git")
            print(f"pushCmd    :\t cd /tmp/gitSync/aws;git rm -r --cached .")
            print(f"copyCmd    :\t cd /tmp/gitSync;cp -r {rootDir} .")
            print(f"rmCacheCmd :\t cd /tmp/gitSync/aws;git add awsCode; git add docs; git add .gitignore;git commit -m {message};git push")
        if not gitAdds:
            raise Exception("specify folders to git add")
        else:
            for gitAdd in gitAdds:
                if gitAdd.startswith(rootDir):
                    raise Exception(f"""
            {gitAdd} should be relative path from rootDir
                eg: {basename(gitAdd)}
                    """)
        sepBy = '; '
        tempRoot = None
        forceAdd = ''
        if cloneCmd:
            # forceAdd = '-f '
            tempRoot = dirop(f'/tmp/{getTimeStamp()}/{basename(rootDir)}')
            gitClone(tempRoot, cloneCmd, skipExe=skipExe)
            for gitAdd in gitAdds:
                src, des = join(rootDir, gitAdd), join(tempRoot, gitAdd)
                if gitAdd == '.gitignore':
                    Path(des).write_text(Path(src).read_text())
                elif gitAdd == 'LICENSE':
                    Path(des).write_text(Path(src).read_text())
                else:
                    dirop(src, cpDir=tempRoot, rm=True)
            rootDir = tempRoot
        if removeCache:
            cmd = f'''
cd {rootDir}
git rm -r --cached .
            '''
            exeIt(cmd, returnOutput=True, sepBy=sepBy, debug=False, skipExe=skipExe)
        gitAdds.insert(0, 'echo')
        gitAdds = f"{sepBy}git add {forceAdd}".join(gitAdds)
        cmd = f'''
cd {rootDir}
{gitAdds}
git commit -a -m "{message}"
git push
    '''
        commitId = 'skipping exe'
        out = exeIt(cmd, returnOutput=True, sepBy=sepBy, debug=True, skipExe=skipExe)
        if out:
            cmd, errCode, out, err = out
            commitId = err.split('..')[-1].split()[0]
            if tempRoot:
                dirop(tempRoot, rm=True, mkdir=False)
        if stop:
            print("787 gitPush pyUtils commitId: ", commitId)
            quit()
        return commitId

    return gitPush, gitPull, gitClone


def getIp(ip, cfgPath=getPath('~/Desktop')):
    if ip == ' ':
        ip = glob(f'{cfgPath}/*.aws*_')
        if len(ip) != 1:
            raise Exception(f"more than one active ip {ip}")
        ip = basename(ip[0]).split('.aws')[0]
        ip = ip.strip().replace('-', '.')
    return ip


def __decodePath(remoteSrc):
    res = ''
    for r in remoteSrc.split('\n'):
        r = r.strip()
        if not r.startswith('#'):
            res += r
    return res


def ec2pull(remoteSrc, desDir, useZip, userName='ubuntu', ip=None, keyPath=getPath('~/.ssh/vishnu.pem')):
    res, desDir = __decodePath(remoteSrc), __decodePath(desDir)
    remoteSrc = res
    ip = ip or getIp(' ')
    remoteSrc, desDir = remoteSrc.strip(), desDir.strip()
    remoteDir = dirname(remoteSrc)
    remoteId = f"{userName}@{ip}"
    rBookName = basename(remoteSrc)
    if useZip:
        bookName = f"tempZip_{getTimeStamp()}.tar.gz"
        cmd = f'''
        echo step 1: zipping remote file/dir
        ssh -i "{keyPath}" {remoteId} "cd {remoteDir};tar -czvf {bookName} {rBookName} > /dev/null 2>&1;echo;echo;echo;ls -lah {bookName}; echo;echo;echo;"
        echo step 2: pulling
        mkdir -p {desDir}
        scp -i "{keyPath}" -r {remoteId}:{remoteDir}/{bookName} {desDir}
        echo step 3: unzipping local file/dir
        cd {desDir};tar -xvzf {bookName} --no-overwrite-dir > /dev/null 2>&1
        echo step 4: deleting remote zip cache
        ssh -i "{keyPath}" {remoteId} "cd {remoteDir};rm -rf {bookName}"
        '''
    else:
        cmd = f'''
        echo step 1: pulling
        mkdir -p {desDir}
        scp -i "{keyPath}" -r {remoteId}:{remoteDir}/{rBookName} {desDir}
        '''
    cmd = f'''
    echo pull {remoteSrc} {desDir}
    echo                               {remoteId.replace('.', '-')}
    echo
    {cmd}
    echo 
    echo
    echo {desDir}    
    '''
    print(cmd)
    print("__________________________________________________")
    with open(dirop('~/aEy22e/cmd/pull.sh', rm=' '), 'w') as book:
        book.write(cmd)


def ec2push(localSrc, desDir, useZip, userName='ubuntu', ip=None, keyPath='~/.ssh/vishnu.pem'):
    ip = ip or getIp(' ')
    localSrc, desDir = localSrc.strip(), desDir.strip()
    zipName = f"tempZip_{getTimeStamp()}.tar.gz"
    remoteId = f"{userName}@{ip}"
    cmd = f'''
    cd {dirname(localSrc)}; tar -czvf {zipName} {basename(localSrc)} > /dev/null 2>&1
    ssh -i "{keyPath}" {remoteId} "mkdir -p {desDir}" 
    scp -i "{keyPath}" -r {zipName} {remoteId}:{desDir}
    ssh -i "{keyPath}" {remoteId} "cd {desDir};tar -xvzf {zipName} --no-overwrite-dir > /dev/null 2>&1;rm -rf {zipName}" 
    '''
    print(cmd)
    print("__________________________________________________")
    with open(dirop('~/aEy22e/cmd/pull.sh'), 'w') as book:
        book.write(cmd)


def demonRunner(logPath, pyPath, mainFn, skipExe=False):
    log_path = dirop(logPath) if logPath else '/dev/null'
    args = getArgs(startDemon=True)
    if args['startDemon']:
        exeCmd = f'cd {dirname(pyPath)};nohup {python} {basename(pyPath)} {mainFn} > {log_path} 2>&1 &'
        if callable(mainFn):
            exeCmd = f'cd {dirname(pyPath)}; nohup {python} {basename(pyPath)} --startDemon=n > {log_path} 2>&1 &'
        print(f"""
________________________________________________________
        {exeCmd}

        cd {dirname(pyPath)}; {python} {basename(pyPath)} {mainFn}

        echo;tail -200 {log_path}; echo


________________________________________________________""")
        if not skipExe:
            os.system(exeCmd)
    else:
        sys.argv = [s for s in sys.argv if s != '--startDemon=n']
        mainFn()


def getCondaPkgs(pkgName=None, cacheDir='~/miniconda3', rm=''):
    cacheDir = getPath(cacheDir)

    def exeCmd(cmd):
        cachePath = f'{cacheDir}/temp.txt'
        os.system(f'{cmd} > {cachePath}')
        with open(cachePath, 'r') as book:
            lines = book.read().split('\n')
        os.remove(cachePath)
        lines = [line for line in lines if line]
        return lines

    def getPkgs():
        lines = exeCmd('conda env list')
        envs = [os.path.basename(line.split()[-1]) for line in lines[3:]]
        print("envs", envs)
        print("_______________________________________________")
        res = defaultdict(list)
        for env in tqdm(envs):
            print(env)
            lines = exeCmd(f'source $HOME/.bash_profile;eval "$(conda shell.bash hook)";conda activate {env};conda list')
            for line in lines[3:]:
                pkg, version = line.split()[:2]
                if 'pypi' in line:
                    msg = f'pip install {pkg}=={version}'
                else:
                    msg = f'conda install {pkg}=={version}'
                res[pkg].append(msg)
        return res

    pkgs = readPkl(f'{cacheDir}/pkgs.pkl', lambda: getPkgs(), rm=rm)
    if pkgName:
        print('\n\n')
        for v in pkgs[pkgName]:
            print(v)
        print('\n\n')
    else:
        for k, vs in pkgs.items():
            print(f"________________{k}________________")
            print('\n\n')
            for v in vs:
                print(v)
                print()
            print('\n\n')


def shaIt(s):
    return f'{int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % 100000000000000000000000000000000:032}'
