import os
import sys
import shutil
import random
import logging
import argparse
import tempfile
import traceback
import numpy as np
from glob import glob
from os.path import join
from pathlib import Path
from os.path import exists
from os.path import dirname
from os.path import basename
from datetime import datetime as dt
from collections import OrderedDict
from collections import defaultdict

try:
    import cv2
except:
    pass

try:
    from PIL import Image
except:
    pass

python = sys.executable
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, formatter=dict(float=lambda xFormatXpw13: f"{xFormatXpw13:8.4f}"))


def setSeed(seed=4487, setTorch=True):
    # print(f"setting random seed to {seed}")
    cv2.setRNGSeed(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        if setTorch:
            import torch
            torch.manual_seed(seed)
    except:
        print("skip setting torch seed")
    return seed


def getArgs(**kwArgs):
    def str2bool(val):
        if val.lower() in ('yes', 'true', 't', 'y', '1'):
            val = True
        elif val.lower() in ('no', 'false', 'f', 'n', '0'):
            val = False
        else:
            raise Exception(f"""
            unknown datatype: {val}
            expected type: ('yes', 'true', 't', 'y', '1'), ('no', 'false', 'f', 'n', '0')
            """)
        return val

    parser = argparse.ArgumentParser()
    for name, value in kwArgs.items():
        argType = type(value)
        if isinstance(value, bool):
            value = 'yes' if value else 'no'
            argType = str2bool
        parser.add_argument(f"--{name}", default=value, type=argType, help=f" eg: {name}={value}")
    return vars(parser.parse_known_args()[0])


def imResize(img, sizeRC=None, scaleRC=None, interpolation=None):
    interpolation = interpolation or cv2.INTER_LINEAR
    if sizeRC is not None:
        r, c = sizeRC[:2]
    else:
        try:
            dr, dc = scaleRC
        except:
            dr, dc = scaleRC, scaleRC
        r, c = img.shape[:2]
        r, c = r * dr, c * dc
    if interpolation == 'aa':
        img = np.array(Image.fromarray(img).resize((int(c), int(r)), Image.ANTIALIAS))
    else:
        img = cv2.resize(img, (int(c), int(r)), interpolation=interpolation)
    return img


def getPath(p):
    p = f"{p}".strip()
    endWithSlash = p.endswith('/') or p.endswith('\\')
    for fn in [os.path.expandvars, os.path.expanduser, os.path.abspath]:
        p = fn(p)
    if endWithSlash:
        p = f"{p}/"
    return p


def osSysMove(src, des):
    os.system(f"""mv "{src}" "{des}" """)


def moveCopy(src, des, op, isFile, rm):
    assert src != des, f'''
src: {src}
des: {des}
both are same path
    '''
    des = getPath(des)
    desDir = dirname(des)
    if not rm and exists(des):
        raise Exception(f'''Fail des: {des}
                                    already exists delete it or try different name
                            eg: change dirop('{src}', cpDir='{desDir}', rm=False)
                                to     dirop('{src}', cpDir='{desDir}', rm=True)
                                or     dirop('{src}', cpDir='{desDir}', rm=False, desName='newName')
                        ''')
    if isFile:
        if rm and exists(des):
            os.remove(des)
    else:
        if rm and exists(des):
            shutil.rmtree(des, ignore_errors=True)
    mkpath = dirname(des)
    if not exists(mkpath):
        try:
            os.makedirs(mkpath)
        except FileExistsError:
            pass
    return op(src, des)


def dirop(path, *, mkdir=True, rm=False, isFile=None, cpDir=None, mvDir=None, symDir=None, desName='', skipExe=False):
    path = getPath(path)
    desName = basename(str(desName) or path)
    if isFile is None:
        isFile = os.path.isfile(path) if os.path.exists(path) else os.path.splitext(path)[-1]
    if skipExe:
        op, des = ['copying', f"{cpDir}/{desName}"] if cpDir else \
            ['moving', f"{mvDir}/{desName}"] if mvDir else \
                ['create shortcut', f"{symDir}/{desName}"] if symDir else \
                    [None, None]
        if op:
            print(f"""
path  : [{'file' if isFile else 'folder'}]\t{path}
des   : {des}
mkdir : {'create desDir if not exists' if mkdir else 'raise error if desDir not exists'}
rm    : {'remove des if exists' if rm else 'raise error if des exists'}
{op}  : {path} -> {des}
            """)
        else:
            print(f"""
path  : [{'file' if isFile else 'folder'}]{path}
mkdir : {'create pathDir if not exists' if mkdir else 'raise error if pathDir not exists'}
rm    : {'remove path if exists' if rm else 'raise error if path exists'}
            """)
        return
    if cpDir or mvDir or symDir:
        if not exists(path):
            raise Exception(f'''Fail src: {path}
                                            not found''')
    elif rm and exists(path):
        if isFile:
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
    mkpath = dirname(path) if isFile else path
    if mkdir and not exists(mkpath) and mkpath:
        try:
            os.makedirs(mkpath)
        except FileExistsError:
            pass
    if cpDir:
        copy = shutil.copy if isFile else shutil.copytree
        path = moveCopy(path, f"{cpDir}/{desName}", copy, isFile, rm=rm)
    elif mvDir:
        desName = desName or path
        path = moveCopy(path, f"{mvDir}/{desName}", osSysMove, isFile, rm=rm)
    elif symDir:
        desName = desName or path
        path = moveCopy(path, f"{symDir}/{desName}", os.symlink, isFile, rm=rm)
    return path


def getTimeStamp():
    return dt.now().strftime("%b%d_%H_%M_%S%f")


def videoPlayer(vpath, startSec=0.0, stopSec=np.inf):
    cam = vpath if type(vpath) == cv2.VideoCapture else cv2.VideoCapture(vpath)
    ok, ftm, fno = True, startSec, 0
    if ftm:
        cam.set(cv2.CAP_PROP_POS_MSEC, ftm * 1000)
    while ok:
        ok, img = cam.read()
        ok = ok and img is not None and ftm < stopSec
        if ok:
            ftm = round(cam.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
            yield fno, ftm, img
            fno += 1


def rglob(p):
    p = getPath(p)
    return glob(p, recursive='**' in p)
    # ps = p.split('**')
    # roots, ps = ps[0], ps[1:]
    # if not ps:
    #     return glob(roots)
    # else:
    #     ps = '**' + '**'.join(ps)
    #     res = []
    #     for root in glob(roots):
    #         for p in Path(root).glob(ps):
    #             res.append(str(p))
    #     return res


def getTraceBack(searchPys, tracebackData=None):
    errorTraceBooks = [basename(p) for p in searchPys or []]
    oTrace = tracebackData or traceback.format_exc()
    trace = oTrace.strip().split('\n')
    msg = trace[-1]
    flow = ''
    for oLine in trace:
        line = oLine.strip()
        if line.startswith('File "'):
            line = line[6:].split('", line')[0]
            if not errorTraceBooks or basename(line) in errorTraceBooks:
                flow = f"\n{oLine}"
    traces = f"""
{oTrace}

{flow}

{msg}    
"""
    return msg, traces


def filename(path):
    return os.path.splitext(basename(path))[0]


def pathname(path):
    return os.path.splitext(path)[0]


def groupBy2(items, key, returnList, *, nSample=None, sortBy=None):
    res = defaultdict(list)
    for item in items:
        keyName = key(item)
        if not (nSample and len(res[keyName]) == nSample):
            res[keyName].append(item)
    if sortBy is not None:
        res = {k: sorted(v, key=sortBy) for k, v in res.items()}
    if returnList:
        res = list(res.items())
    return res
