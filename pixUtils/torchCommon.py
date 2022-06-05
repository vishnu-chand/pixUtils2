import math
import torch
import random
from torch import nn
from pixUtils import *
from copy import deepcopy
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except:
    pass
try:
    from skimage.filters import gaussian
except:
    pass
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, sci_mode=False)


def Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0, reverse=False):
    if reverse:
        stdR = np.reciprocal(std)
        meanR = -np.array(mean) * stdR
        return A.Normalize(mean=meanR, std=stdR, max_pixel_value=1.0, always_apply=always_apply, p=p)
    else:
        return A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)


def img2torch(x, device, normalize=None):
    normalize = normalize or Normalize()
    if type(x) == list:
        x = np.array(x)
    x = normalize(image=x)['image']
    x = torch.from_numpy(x).to(device)
    x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)
    return x


def mask2torch(x, device):
    if type(x) == list:
        x = np.array(x)
    x = torch.from_numpy(x)
    nCh = len(x.shape)
    if nCh == 3:
        x = torch.unsqueeze(x, dim=1)
    else:
        x = x[None]
    return x.to(device)


def torch2img(x, normalize=None, float2img=True):
    normalize = normalize or Normalize(reverse=True)
    nCh = len(x.shape)
    if nCh == 4:
        x = x.permute(0, 2, 3, 1)
    if nCh == 3:
        x = x.permute(1, 2, 0)
    x = x.detach().cpu().numpy()
    if normalize:
        x = normalize(image=x)['image']
    if float2img:
        x = np.uint8(255 * x)
    return x


def torch2mask(x, dtype='f4'):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        if dtype is not None:
            x = x.astype(dtype)
    return x


def ckpt2pth(ckptPath, pthPath, model=None, lModel=None):
    if exists(pthPath):
        raise Exception(f"already exists: {pthPath}")
    if lModel is None:
        class DummyLit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

        lModel = DummyLit()
    ckpt = torch.load(ckptPath, map_location='cpu')
    lModel.load_state_dict(ckpt['state_dict'], strict=True)
    torch.save(lModel.model.state_dict(), pthPath)
    print(f"""conversion of 
            {ckptPath}
             to 
             {pthPath}
            completed
                """)


def setLearningRate(model, lr2name, verbose=True):
    '''
lr2name = defaultdict(list)
for n, w in model.named_parameters():
    print(n)
    lr2name[0.01].append(n)
print("________________________________________")
for n, m in model.named_modules():
    a = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
         nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,
         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    if isinstance(m, a):
        print(n)
        lr2name['freezeBN'].append(n)
model, lrParams = setLearningRate(model, lr2name=lr2name, verbose=True)
optimizer = optim.Adam(lrParams)
    '''
    if verbose:
        for lr, name in lr2name.items():
            print(lr, name)

    name2lr = {}
    for lr, ns in lr2name.items():
        for n in ns:
            name2lr[n] = lr

    lrParams = [name2lr[n] for n, w in model.named_modules() if name2lr.get(n) is not None]
    if lrParams:
        for moduleName, module in model.named_modules():
            if name2lr.get(moduleName, '').lower() == 'freezebn':
                module.eval()
                module.requires_grad_(False)

    lrParams = [name2lr[n] for n, w in model.named_parameters() if name2lr.get(n) is not None]
    if lrParams:
        lrParams = [dict(params=w, lr=name2lr[n]) for n, w in model.named_parameters()]
    return model, lrParams


def weight_init(m):
    import torch.nn.init as init
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def loadWeights(model, weights, layerMap=None, debug=None, applyWeightInit=False):
    '''
    loadWeights(net_g, torch.load(gPath, map_location='cpu')['model'], layerMap={'emb_g.weight': dict(lName='emb_g.weight', axis0=np.arange(newNSpeakers) % pretrainedNSpeakers, axis0Rand=[-1, -2])})
    '''
    if applyWeightInit:
        model.apply(weight_init)
    if not isinstance(weights, OrderedDict):
        weights = weights.state_dict()
    if debug:
        print(pd.DataFrame([(k, v.shape, v.dtype) for k, v in weights.items()]))

    layerMap = layerMap or dict()
    newWeights = OrderedDict()
    for lName, lWeight in weights.items():
        mapData = layerMap.get(lName)
        if mapData:
            print(f"modifying: {lName} [{lWeight.shape}]", end=' ')
            lName, axis0, axis1, axis0Rand, axis1Rand = mapData.get('lName'), mapData.get('axis0', []), mapData.get('axis1', []), mapData.get('axis0Rand', []), mapData.get('axis1Rand', [])  # name
            if len(axis0):
                lWeight = lWeight[axis0]  # axis0
            if len(axis0Rand):
                lWeight[axis0Rand] = torch.nn.init.xavier_uniform_(lWeight[axis0Rand])  # axis0
            if len(axis1):
                lWeight = lWeight[:, axis1]  # axis1
            if len(axis1Rand):
                lWeight[:, axis1Rand] = torch.nn.init.xavier_uniform(lWeight[:, axis1Rand])  # axis1
            print(f"to {lName} [{lWeight.shape}]")

        newWeights[lName] = lWeight

    if debug:
        print(pd.DataFrame([(k, v.shape, v.dtype) for k, v in newWeights.items()]))
    print("_________________________ loading weights _________________________")
    msg = model.load_state_dict(newWeights, strict=False)
    data = [f"'{d}'," for d in model.state_dict() if d not in msg.missing_keys]
    nData = len(data)
    if debug:
        data = '\n'.join(data)
        print(f"ok = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.missing_keys])
    print(f"model = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.unexpected_keys])
    print(f"weight = [\n{data}\n]")
    print("nWeightLoaded: ", nData)
    print("nModelMiss   : ", len(msg.missing_keys))
    print("nWeightMiss  : ", len(msg.unexpected_keys))
    if debug:
        print("195 loadWeights torchCommon : ", );
        quit()


def describeOptimizer(optimizerStateDict):
    assert isinstance(optimizerStateDict, dict), "input should be state dict"
    for k, kv in optimizerStateDict['state'].items():
        for s, v in kv.items():
            try:
                prr(f"optimizerStateDict['state']['{k}']['{s}']: ", v)
            except:
                print(k, v)
    print("***********************************************")
    for k, v in optimizerStateDict['param_groups'][0].items():
        try:
            prr(f"38 load_state_dict optimizer {k}: v", v)
        except:
            print(k, v)


def describeModel(model, inputs, batchSize, device=device, summary=True, fps=False, remove=(), *a, **kw):
    assert type(inputs) == list
    cols = ("Index", "Type", "Channels", "Kernel Shape", "Output Shape", "Params", "Mul Add")
    remove = [x.lower().replace(' ', '') for x in remove]
    inputs = [x.to(device) for x in inputs]

    cols = [x for x in cols if x.lower().replace(' ', '') not in remove]
    removeIndex = False if "Index" in cols else True
    if not removeIndex:
        cols.pop(0)

    def getSummary(model, inputs, *args, **kwargs):
        def get_names_dict(model):
            """Recursive walk to get names including path."""
            names, types = dict(), dict()

            def _get_names(module, parent_name=""):
                def decodeKey(key):
                    try:
                        int(key)
                        key = f"[{key}]"
                        # key = f".{key}"
                    except:
                        key = f".{key}"
                    return key

                for key, m in module.named_children():
                    cls_name = str(m.__class__).split(".")[-1].split("'")[0]
                    num_named_children = len(list(m.named_children()))
                    name = f"{parent_name}{decodeKey(key)}" if parent_name else key
                    names[name] = m
                    types[name] = cls_name

                    if isinstance(m, torch.nn.Module):
                        _get_names(m, parent_name=name)

            _get_names(model)
            return names, types

        def register_hook(module):
            # ignore Sequential and ModuleList
            if not module._modules:
                hooks.append(module.register_forward_hook(hook))

        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mul Add in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        module_names, types = get_names_dict(model)

        hooks = []
        summary = OrderedDict()

        model.apply(register_hook)
        try:
            with torch.no_grad():
                model(*inputs)
        finally:
            for hook in hooks:
                hook.remove()

        # Use pandas to align the columns
        df = pd.DataFrame(summary).T
        types = [types['_'.join(name.split('_')[1:])] for name in df.index]
        df['Type'] = np.array(types).T
        df["Mul Add"] = pd.to_numeric(df["macs"], errors="coerce")
        df["Params"] = pd.to_numeric(df["params"], errors="coerce")
        df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
        # inData = [list(x.shape)] + df.out.tolist()[:-1]
        df = df.rename(columns=dict(ksize="Kernel Shape", out="Output Shape"))
        # df["Input Shape"] = inData
        df["Channels"] = df[["Output Shape"]].applymap(lambda x: x[1]).to_numpy().tolist()
        df_sum = df.sum()
        df.index.name = "Layer"
        if removeIndex:
            df.index = [''] * len(df)
        df = df[list(cols)]

        option = pd.option_context("display.max_rows", None, "display.max_columns", 10, "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True))
        with option:
            max_repr_width = max([len(row) for row in df.to_string().split("\n")])
            print("=" * max_repr_width)
            print(df.replace(np.nan, "-"))
            print("-" * max_repr_width)
            df_total = pd.DataFrame(
                    {"Total params"        : (df_sum["Params"] + df_sum["params_nt"]),
                     "Trainable params"    : df_sum["Params"],
                     "Non-trainable params": df_sum["params_nt"],
                     "Mul Add"             : df_sum["Mul Add"]
                     },
                    index=['Totals']
            ).T
            print(df_total)
            print("=" * max_repr_width)
        return df

    if device is not None:
        model = model.to(device)
    print("model", type(model))
    df = None
    if summary:
        df = getSummary(model, inputs, *a, **kw)
    if fps:
        for i in range(5):
            model(*inputs)
        nIter = 15
        tik = clk()
        for i in range(nIter):
            model(*inputs)
        tok = tik.tok("").last()
        nFrame = nIter * batchSize
        fps = nFrame / tok
        print(f"""
fps   : {fps:9.6f}
ms    : {1 / fps:9.6f}""")
    return df


def applyTransforms(transformers, data):
    for transformer in transformers:
        if type(transformer) == list:
            data = applyTransforms(transformer, data)
        else:
            data.update(transformer(**data))
    return data


class GenImgData(Dataset):
    def __init__(self, readers, transformers, returnKeys):
        i = 0
        res = []
        for t, n in readers:
            res.append([t, i, i + n])
            i += n
        self.t = res
        self.nItems = i
        self.transformers = transformers
        self.returnKeys = returnKeys

    def __len__(self):
        return self.nItems

    def __getitem__(self, ix):
        for i in range(10):
            data = self.doTransform(ix)
            if data['ok']:
                data = {k: data[k] for k in self.returnKeys}
                return data
            else:
                ix = random.randint(0, self.nItems - 1)
        data = self.doTransform(ix)
        return {k: data[k] for k in self.returnKeys}

    def doTransform(self, ix):
        for t, s, e in self.t:
            if ix < e:
                data = t(ix - s)
                break
        else:
            raise
        data = applyTransforms(self.transformers, data=data)
        return data


def image_copy_paste(img, paste_img, alpha, alphaWeight, blend=True, sigma=1):
    img_dtype = img.dtype
    if len(alpha.shape) == 3:
        alpha = alpha.max(axis=-1)
    alpha = np.float32(alpha != 0)
    if blend:
        alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
    if len(img.shape) == 3:
        alpha = alpha[..., None]
    alpha *= alphaWeight
    img = paste_img * alpha + img * (1 - alpha)
    img = img.astype(img_dtype)
    return img


def CopyPaste(srcFn, pasteFn, pasteImPaths, minOverlay=.7, p=0.5):
    def __copyPaste(**data):
        src = applyTransforms(srcFn, data)
        image, mask = src['image'], src['mask']
        if random.random() < p:
            f = cv2.imread(random.choice(pasteImPaths), cv2.IMREAD_UNCHANGED)
            data = dict(image=f[..., :3], mask=f[..., 3])
            paste = applyTransforms(pasteFn, data)
            pasteImage, pasteMask = paste['image'], paste['mask']
            f = pasteMask.astype('f4') / 255
            image = image_copy_paste(image, pasteImage, f, blend=True, alphaWeight=min(1.0, minOverlay + random.random()))
            mask = image_copy_paste(mask, pasteMask, f, blend=False, alphaWeight=1)
        data['image'], data['mask'] = image, mask
        return data

    return __copyPaste


def getDevice(x):
    return torch.device('cuda') if x.is_cuda else torch.device('cpu')


def asDevice(x, y):
    if x.is_cuda != y.is_cuda:
        x, y = x.to('cuda'), y.to('cuda')
    return x, y


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2
    https://fastai.github.io/timmdocs/training_modelEMA
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        with torch.no_grad():
            super(ModelEmaV2, self).__init__()
            # make a copy of the model for accumulating moving average of weights
            self.module = model
            self.module.eval()
            self.decay = decay
            self.device = device  # perform ema on different device from model if set
            if self.device is not None:
                self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class TopK:
    '''
    topK = TopK(10, .5, 'train')
    xs = np.arange(25).reshape(5, 5)
    for ix, _ in enumerate(range(100), 1):
        print("___________________________")
        for x in xs:
            batchLoss = random.random() / ix
            topK.forward(x, batchLoss, None)
        print(topK)
    TopK.readLog('/home/ubuntu/aEy22e/girishTorchFS2v2/trainDec03_19_40_17023588.log')
    '''

    def __init__(self, topK=50, histWeight=0.5, trainVal='train'):
        self.topK = topK
        self.data = dict()
        self.lossTh = -np.inf
        self.trainVal = trainVal
        self.histWeight = histWeight

    def forward(self, ids, batchLoss, data):
        try:
            batchLoss = batchLoss.item() / len(ids)  # avg loss per batch
        except:
            batchLoss = np.array(batchLoss) / len(ids)
        updated = False
        for ix in ids:
            if len(self.data) < self.topK or ix in self.data or batchLoss >= self.lossTh:
                updated = True
                crntLoss = batchLoss
                if ix in self.data:
                    crntLoss = self.data[ix]['loss'] * self.histWeight + crntLoss * (1 - self.histWeight)
                self.data[ix] = dict(ix=ix, loss=crntLoss, data=data)
        if updated:
            data = sorted(self.data.items(), key=lambda x: x[1]['loss'], reverse=True)
            data = data[:self.topK]
            self.lossTh = data[-1][1]['loss']  # last element in topK is the entry threshold
            self.data = dict(data)

    def __repr__(self):
        data = {k: v['loss'] for k, v in self.data.items()}
        res = f'\ntop {self.trainVal} error:\t {data}'
        # for ix, (k, v) in enumerate(sorted(self.data.items(), key=lambda x: x[1]['loss'])):
        #     res += f"[{ix:03d}]: {k} : {v['loss']:8.4f}, \t"
        return res

    @staticmethod
    def readLog(logPath):
        with open(logPath, 'r') as book:
            lines = book.read().split('\n')
        topK = defaultdict(list)
        for line in lines[::-1]:
            if len(topK) > 50:
                break
            if '=' in line:
                line = line.replace('=', ',').split(',')
                line = [l.strip() for l in line]
                f = line[0]
                try:
                    losses = [float(l) for l in line[1:]]
                    topK[f].extend(losses)
                except:
                    pass
        topK = sorted(topK.items(), key=lambda x: np.array(x[1]).sum())
        for k, v in topK:
            v = np.array(v)
            print(k)


class LossLog:
    def __init__(self, logPath='', cols=''):
        self.logPath = open(logPath, 'a')
        self.logPath.write(f"{','.join(cols)}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logPath.close()

    def forward(self, keys, stepEpoch, losses):
        data = [str(stepEpoch)]
        for loss in losses:
            try:
                loss = loss.item()
            except:
                pass
            data.append(str(round(loss, 3)))
        data.append(' '.join(keys))
        self.logPath.write(f"{','.join(data)}\n")


class SequentialLR(_LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                        "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                        "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                    "Sequential Schedulers expects number of schedulers provided to be one more "
                    "than the number of milestone points, but got number of schedulers {} and the "
                    "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1

    def step(self):
        from bisect import bisect_right
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.005    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                               (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                           (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=1.25, max_lr=0.002, min_lr=0.00001, warmup_steps=10, gamma=0.75, last_epoch=-1)
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    @staticmethod
    def testIt(epochs, first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma, last_epoch):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
        lr_scheduler_1 = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=cycle_mult, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=gamma, last_epoch=last_epoch)
        lrs = []
        for i in range(epochs):
            lr_scheduler_1.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        for lr in lrs:
            print(lr)
        plt.plot(lrs)
        plt.show()


def lossLogT2est():
    from pydub.playback import play
    from pydub import AudioSegment
    logPaths = rglob('/home/ubuntu/aEy22e/vc6v2/consolidateLog/*.log')
    for logPath in logPaths:
        df = pd.read_csv(logPath)[5000:]
        df.wavPath = df.wavPath.str.replace('/home/ubuntu/aEye/db/vits/Data', '/home/ubuntu/aEye/db/v2/ready2train')
        df = df[df.wavPath.str.find('vctkPre') == -1]
        df = df.groupby('wavPath').mean()
        loss = 'loss_mel  loss_disc  loss_dur  loss_kl  loss_fm'.split()
        df = df[loss]
        df[df < df.quantile(.98)] = np.nan
        print("810 lossLogT2est torchCommon df.size: ", df.size)
        df = pd.DataFrame(df.dropna(how='all'))
        # odf = df.fillna(-1)
        print(df)
    # for col in loss:
    #     for i in range(50):
    #         print("816 lossLogT2est torchCommon col: ", col)
    #     df = odf.copy().sort_values(col, ascending=False)
    #     for wavPath, r in df.head().iterrows():
    #         if exists(wavPath):
    #             print(readBook(f"{pathname(wavPath)}.txt"))
    #             play(AudioSegment.from_file(wavPath))
    #             input(1)
    #         else:
    #             print(f"skipping {wavPath}")
    # print("814 lossLogT2est torchCommon df.size: ", df.size)


if __name__ == '__main__':
    # CosineAnnealingWarmupRestarts.testIt(500, first_cycle_steps=50, cycle_mult=1.25, max_lr=0.0002, min_lr=0.00001, warmup_steps=10, gamma=0.75, last_epoch=-1)
    lossLogT2est()
