from pixUtils import *
try:
    import textgrid
except:
    pass
from pydub import AudioSegment
from pydub.playback import play


def convertAudio(src, des):
    wav = AudioSegment.from_file(src)
    ext = os.path.splitext(des)[-1][1:]
    wav.export(des, ext)


def readTextGrid(textGridPath):
    text_grid, phone_grid = textgrid.TextGrid.fromFile(textGridPath)
    text_grid = [(int(i.minTime * 1000), int(i.maxTime * 1000), i.mark) for i in text_grid]  # time in milli sec
    phone_grid = [(int(i.minTime * 1000), int(i.maxTime * 1000), i.mark) for i in phone_grid]  # time in milli sec
    return text_grid, phone_grid


def getLibriData(dataPaths, path2name):
    assert type(dataPaths) == list, [type(dataPaths)]
    wavData = defaultdict(list)
    for wavPath in tqdm(dataPaths):
        sid = path2name(wavPath)
        wavData[sid].append(wavPath)
    return sorted(wavData.items(), key=lambda x: len(x[1]), reverse=True)


def generateTextGrid(inDir, outDir, failDir, *, lexicon='lexicon/librispeech-lexicon.txt', acoustic_model_path='montreal-forced-aligner/pretrained_models/english.zip', nJob=8):
    def moveFailure(wavPath, failDir):
        if wavPath:
            print(f"failed paring {wavPath} mv to {failDir}")
            assert len(wavPath) == 1, f"""
matched more than one text paths
{wavPath}
"""
            wavPath = wavPath[0]
            dirop(wavPath, mvDir=failDir)
            textPath = rglob(f"{pathname(wavPath)}*txt") + rglob(f"{pathname(wavPath)}*lab")
            assert len(textPath) == 1, f"""
matched more than one text paths
{textPath}
"""
            dirop(textPath[0], mvDir=failDir)

    def removeNonAligned(inDir, outDir, failDir):
        ok = True
        if exists(f"{outDir}/unaligned.txt"):
            ok = False
            with open(f"{outDir}/unaligned.txt", 'r') as book:
                lines = book.read().split('\n')
            wavPaths = [rglob(f"{inDir}/**/{line.split()[0]}.wav") for line in lines if line]
            for wavPath in wavPaths:
                moveFailure(wavPath, f"{failDir}/nonAlign")
        return ok

    def removeOOVS(inDir, outDir, failDir):
        ok = True
        for i in rglob(f"{outDir}/**/*.TextGrid"):
            ok = False
            text_grid, phone_grid = readTextGrid(i)
            for unknown, grid in [['<unk>', text_grid], ['spn', phone_grid]]:
                for s, e, w in text_grid:
                    if w == unknown:
                        print(' '.join([w for s, e, w in text_grid]))
                        wavPath = rglob(f"{inDir}/**/{filename(i)}.wav")
                        moveFailure(wavPath, f"{failDir}/unknown")
        return ok

    inDir, outDir = getPath(inDir), getPath(outDir)
    mfaRoot = '~/Documents/mfa'
    os.system('cd ~/Documents;rm -rf mfa MFA; cp -r ~/aEye/bkDb/mfa .')
    exeIt(f"cd {mfaRoot};./montreal-forced-aligner/bin/mfa_align {inDir} {lexicon} {acoustic_model_path} {outDir} -j {nJob}", returnOutput='', skipExe='')
    alignOk, oovsOk = True, True
    if failDir:
        failDir = getPath(failDir)
        alignOk = removeNonAligned(inDir, outDir, failDir)
        oovsOk = removeOOVS(inDir, outDir, failDir)
    print(f'''

inDir  :   {inDir}
outDir :   {outDir}
failDir:   {failDir}
    ''')
    if not (alignOk or oovsOk):
        raise


def dispSample(root, wavPath):
    root = getPath(root)
    wavPath = getPath(wavPath)
    import textgrid
    from text import text_to_sequence
    # from utils.model import get_vocoder
    # from audioAugmentation import np2audioSegment
    wavDir = dirname(wavPath)
    fName = filename(wavPath)
    metas = Path(rglob(f'{root}/**/train.txt')[0]).read_text() + Path(rglob(f'{root}/**/val.txt')[0]).read_text()
    metaData = dict()
    for meta in metas.splitlines():
        metaName, _, phone, txt = meta.split('|')
        metaData[metaName] = metaName, _, phone, txt
    metaName, personId, phone, txt = metaData[fName]
    speakers = json.load(open(f'{root}/preprocessed_data/LibriTTS/speakers.json'))
    stats = json.load(open(f'{root}/preprocessed_data/LibriTTS/stats.json'))
    normalizedTxt = Path(f"{wavDir}/{fName}.normalized.txt").read_text()
    try:
        originalTxt = Path(f"{wavDir}/{fName}.original.txt").read_text()
    except:
        originalTxt = 'not found'
    labTxt = Path(rglob(f"{root}/**/{fName}.lab")[0]).read_text()
    txtGrid = Path(rglob(f'{root}/**/{fName}.TextGrid')[0])
    mel = np.load(rglob(f'{root}/**/mel/*{fName}.npy')[0])
    pitch = np.load(rglob(f'{root}/**/pitch/*{fName}.npy')[0])
    energy = np.load(rglob(f'{root}/**/energy/*{fName}.npy')[0])
    duration = np.load(rglob(f'{root}/**/duration/*{fName}.npy')[0])
    txtGrid = textgrid.TextGrid.fromFile(txtGrid)
    txtGridWords = [(int(i.minTime * 1000), int(i.maxTime * 1000), i.mark) for i in txtGrid[0]]  # time in milli sec
    txtGridPhones = [(int(i.minTime * 1000), int(i.maxTime * 1000), i.mark) for i in txtGrid[1]]  # time in milli sec
    print("metaName         :", f"{personId} {metaName}")
    print("normalizedTxt    :", normalizedTxt)
    print("originalTxt      :", originalTxt)
    print("labTxt           :", labTxt)
    print("txt              :", txt)
    print("encodedTxt       :", text_to_sequence(normalizedTxt, ['english_cleaners']))
    print("phone            :", phone)
    print("txtGridWords     :", txtGridWords)
    print("txtGridPhones    :", txtGridPhones)
    print("wavPath          :", wavPath)
    print("speakers         :", speakers)
    print("stats            :", stats)
    prr("mel", mel)
    prr("pitch", pitch)
    prr("energy", energy)
    prr("duration", duration)
    # wav = AudioSegment.from_file(wavPath)
    # play(wav)
    # vocoder = get_vocoder(yaml.load(open('config/LibriTTS/model.yaml', "r"), Loader=yaml.FullLoader), device='cpu', hifiGanRoot=getPath('~/aEye/db/fastSpeech2/hifigan1'))
    # transpose = torch.from_numpy(mel)[None,].transpose(2, 1)
    # wavNp = vocoder(transpose)[0].detach().numpy() * 32768
    # play(np2audioSegment(wavNp, frame_rate=22050, channels=1))


class Text2machineSeq:
    # from g2p_en import G2p
    try:
        from text import symbols
        from string import punctuation
        from text import text_to_sequence
    except:
        print(f"fail to import text ")
    # g2p = G2p()

    def __init__(self, lexicon_path, text_cleaners, useG2p=True):
        self.useG2p = useG2p
        self.lexicon = self.read_lexicon(lexicon_path)
        self.text_cleaners = text_cleaners

    @staticmethod
    def getSymbols():
        print(f"nText2machineSeq.symbols: {len(Text2machineSeq.symbols)}")
        return Text2machineSeq.symbols

    @staticmethod
    def read_lexicon(lex_path):
        lexicon = {}
        with open(getPath(lex_path)) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def text2seq(self, text, verbose=False):
        useG2p = self.useG2p
        lexicon = self.lexicon
        g2p = self.g2p
        phones = []
        text = text.rstrip(Text2machineSeq.punctuation)
        if useG2p:
            words = re.split(r"([,;.\-\?\!\s+])", text)
            for w in words:
                if w.lower() in lexicon:
                    phones += lexicon[w.lower()]
                else:
                    phones += list(filter(lambda p: p != " ", g2p(w)))
            phones = "{" + "}{".join(phones) + "}"
            phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
            phones = phones.replace("}{", " ")
        else:
            phones = text
        sequence = np.array(Text2machineSeq.text_to_sequence(phones, self.text_cleaners))
        if verbose:
            print(f"Raw Text Sequence [{int(useG2p)}]: {text}")
            print(f"Phoneme Sequence  [{int(useG2p)}]: {phones}")
            print(f"Machine Language  [{int(useG2p)}]: {sequence}")
            if useG2p:
                phoneList = replaces(phones, '{', '', '}', '').split()
                assert len(phoneList) == len(sequence)
                p2s = [f'{p}-{s}' for p, s in zip(phoneList, sequence)]
                print(f"phone2machine     [{int(useG2p)}]: {' '.join(p2s)}")
        return sequence


def audioSegment2np(clip):
    return np.array(clip.get_array_of_samples()).astype('f4')


def np2audioSegment(npWav, *, wav=None, frame_rate=None, channels=None):
    if wav is not None:
        frame_rate = wav.frame_rate
        channels = wav.channels
    npWav = npWav.astype('i2')
    audio_segment = AudioSegment(npWav.tobytes(), frame_rate=frame_rate, sample_width=npWav.dtype.itemsize, channels=channels)
    return audio_segment


def playAudio(wav, refWav=None):
    import torch
    if isinstance(refWav, str):
        refWav = AudioSegment.from_file(refWav)
    if isinstance(wav, np.ndarray):
        wav = np2audioSegment(npWav=wav, wav=refWav)
    elif isinstance(wav, torch.Tensor):
        wav = np2audioSegment(npWav=wav.detach().cpu().numpy(), wav=refWav)
    play(wav)


def googleTTS(desPath, text, voice, enabledVoices, debug, speed=1.0, pitch=0.0, rm=False):
    assert voice in enabledVoices
    if rm or not exists(desPath):
        try:
            desPath = dirop(desPath)
            cmd = f''' 
        curl -X POST "http://videogroup-env.eba-7pwcun4b.us-east-1.elasticbeanstalk.com:80/GenerateVoice" 
        -d "text={text}&voice={voice}&speed={speed}&pitch={pitch}&generate_unique=true" --max-time 60
        '''
            cmd, errCode, out, err = exeIt(cmd, returnOutput=' ', debug=debug)
            out = ast.literal_eval(out)['filename']
            cmd = f'''
        curl http://videogroup-env.eba-7pwcun4b.us-east-1.elasticbeanstalk.com/{out} -o {desPath}
        '''
            exeIt(cmd, returnOutput=' ', debug=debug)
            wav = AudioSegment.from_file(desPath).set_channels(1).set_frame_rate(22050)
            wav.export(desPath, "wav")
            Path(f"{pathname(desPath)}.txt").write_text(text)
        except Exception as exp:
            raise Exception(f"""
{exp}
text: {text}
            """)


def googleSTT(wavPath, gsDes, language_code, keyPath):
    from google.cloud import speech_v1 as speech
    if not wavPath.endswith('.flac'):
        wav = AudioSegment.from_file(wavPath)
        wavPath = f"{pathname(wavPath)}_stt.flac"
        wav.set_channels(1).export(wavPath, 'flac')
    exeIt(f'gsutil cp {wavPath} {gsDes}', debug=False)
    config = dict(language_code=language_code, enable_automatic_punctuation=True, enable_word_time_offsets=True, enable_separate_recognition_per_channel=True, use_enhanced=True, model='video', enable_word_confidence=True, enable_spoken_punctuation=True)
    client = speech.SpeechClient.from_service_account_file(getPath(keyPath))
    response = client.long_running_recognize(config=config, audio=dict(uri=gsDes))
    datas = []
    for result in response.result().results:
        data = []
        for word in result.alternatives[0].words:
            start_ms = word.start_time.total_seconds() * 1000
            end_ms = word.end_time.total_seconds() * 1000
            data.append([round(start_ms), round(end_ms), word.confidence, word.word])
        if data:
            datas.extend(data)
    exeIt(f'gsutil rm {gsDes}', debug=False)
    return datas
