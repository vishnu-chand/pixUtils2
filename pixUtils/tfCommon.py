import h5py
import tensorflow as tf
from pixUtils import *


def getH5KeyVal(h5weight, read):
    assert read in ['file', 'folder', 'emptyFolder', 'all']

    class ReadH5:
        def __init__(self, read=None):
            self.names = []
            self.read = read

        def __call__(self, name, h5obj):
            if self.read == 'file':
                if isinstance(h5obj, h5py.Dataset) and name not in self.names:
                    self.names.append(name)
            elif self.read == 'folder':
                if isinstance(h5obj, h5py.Group) and name not in self.names:
                    self.names.append(name)
            elif self.read == 'emptyFolder':
                if isinstance(h5obj, h5py.Group) and name not in self.names and not h5obj.values():
                    self.names.append(name)
            elif self.read == 'all':
                self.names.append(name)
            else:
                raise
    if type(h5weight) == str:
        with h5py.File(h5weight) as h5weight:
            reader = ReadH5(read=read)
            h5weight.visititems(reader)
            names = reader.names
    else:
        reader = ReadH5(read=read)
        h5weight.visititems(reader)
        names = reader.names
    return names


def loadTfWeight(model, wpath, layerMap=None, displayLayers=False):
    wpath = getPath(wpath)
    if displayLayers:
        model.summary()
        mData = OrderedDict()
        for layer in model.layers:
            for weight in layer.weights:
                mData[f"{layer.name}/{weight.name}"] = weight
        for m in mData:
            print(m)
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        with h5py.File(wpath) as wDatas:
            for lName in getH5KeyVal(wDatas, 'file'):
                print(lName)
        print("121 loadTfWeight zcreateDbTensorTTS : ", );quit()
    print("____________________________________________________")
    if layerMap is not None:
        mData = {f"{layer.name}/{weight.name}": weight for layer in model.layers for weight in layer.weights}
        with h5py.File(wpath) as wDatas:
            layerWeights = dict()
            for lName in getH5KeyVal(wDatas, 'file'):
                wData = np.array(wDatas[lName])
                modify = layerMap.get(lName)
                if not modify:
                    layerWeights[lName] = wData
                else:
                    newName = modify['lName']
                    if newName is None:
                        print(f"skip loading h5 : {lName}")
                    else:
                        print(f"modifying h5    : {lName}")
                        lName = newName
                        wData = modify.get('wData', wData)
                        layerWeights[lName] = wData
            wLNames, mLNames = set(layerWeights), set(mData)
            print("missing wLNames =", list(sorted(wLNames - mLNames)))
            print("missing mLNames =", list(sorted(mLNames - wLNames)))
            for lName in mData:
                wData = layerWeights.get(lName)
                if wData is None:
                    print(f"skip loading    : {lName}")
                else:
                    try:
                        mData[lName].assign(wData)
                    except Exception as exp:
                        msg = f"""

{exp}
lName             : {lName}
mData[lName].shape: {mData[lName].shape}
wData[lName].shape: {wData.shape}
                        """
                        raise Exception(msg)
    else:
        model.load_weights(wpath, by_name=False, skip_mismatch=False)
    print(f"loading weights completed: {wpath}")
    print("____________________________________________________")
    return model
