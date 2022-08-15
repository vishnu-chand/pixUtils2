from pixUtils import *


def downloadS3(isDir, src, desDir, desName='', skipIfExists=True, skipExe=False, reverseIt=False, verbose=True, allowLocal=False):
    desName = desName if desName != '' else basename(src)
    des = getPath(f"{desDir}/{desName}")
    if reverseIt:
        des = uploadS3(isDir, src=des, desDir=dirname(src), desName=basename(src), skipExe=skipExe)
    elif not exists(des) or not skipIfExists:
        assert src.startswith('s3://'), f"src should start with 's3://' : {src}"
        if not allowLocal:
            assert not exists(getPath('~/aEye/I_AM_LOCAL_MACHINE')), f"trying to run on local machine, {src}"
        awsCmd = f'aws s3 {"sync" if isDir else "cp"} {src.rstrip("/")} {des.rstrip("/")}'
        if verbose or skipExe:
            print(awsCmd)
        awsKey = f"export AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']};export AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"
        exeIt(f'{awsKey};{awsCmd} > /dev/null 2>&1', debug=False, dispCmd=False, skipExe=skipExe)
    return des


def uploadS3(isDir, src, desDir, desName='', skipExe=False, reverseIt=False, verbose=True, allowLocal=False):
    desName = desName if desName != '' else basename(src)
    des = f"{desDir}/{desName}"
    if reverseIt:
        des = downloadS3(isDir, src=des, desDir=dirname(src), desName=basename(src), skipExe=skipExe)
    else:
        assert exists(src), f"file not found: {src}"
        assert desDir.startswith('s3://'), f"desDir should start with 's3://' : {desDir}"
        if not allowLocal:
            assert not exists(getPath('~/aEye/I_AM_LOCAL_MACHINE')), f"trying to run on local machine"
        awsCmd = f'aws s3 {"sync" if isDir else "cp"} "{src.rstrip("/")}" "{des.rstrip("/")}"'
        if verbose or skipExe:
            print(awsCmd)
        awsKey = f"export AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']};export AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"
        exeIt(f'{awsKey};{awsCmd} > /dev/null 2>&1', debug=False, dispCmd=False, skipExe=skipExe)
    return des


def s3exists(s3Path):
    s3Path = s3Path.rstrip('/')
    awsCmd = f"aws s3 ls '{s3Path}'"
    assert s3Path.startswith('s3://'), f"s3Path should start with 's3://' : {s3Path}"
    awsKey = f"export AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']};export AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"
    out = exeIt(f'{awsKey};{awsCmd}', debug=False, dispCmd=False, raiseOnException=False, returnOutput=True)[2].split('\n')[0]
    out = out[4:-1] if out.endswith('/') else out[31:]
    return out == basename(s3Path)


def s3rm(isDir, s3Path, skipExe=False, verbose=True):
    awsCmd = f'aws s3 rm {"--recursive" if isDir else ""} {s3Path}'
    assert s3Path.startswith('s3://'), f"s3Path should start with 's3://' : {s3Path}"
    awsKey = f"export AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']};export AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"
    if verbose or skipExe:
        print(f"deleting: {s3Path}")
    exeIt(f'{awsKey};{awsCmd} > /dev/null 2>&1', debug=False, dispCmd=False, skipExe=skipExe)
    return s3Path


def rawsGlob(s3path, raiseOnEmpty=True):
    assert s3path.startswith('s3://')
    assert '*' in s3path, f"* missing {s3path}"
    root, path = [], []
    for x in s3path.split(os.sep):
        if not path and '*' not in x:
            root.append(x)
        else:
            path.append(x)
    print(f"""68 rawsGlob s3Utils root, path: {os.sep.join(root), os.sep.join(path)}""")
    cmd, errCode, out, err = exeIt(f'aws s3 sync {os.sep.join(root)} . --exclude "*" --include "{os.sep.join(path)}" --dryrun', debug=False)
    if raiseOnEmpty:
        assert out, f"file not found: {s3path}"
    return [x.split('download: ')[1].split(' to ')[0] for x in out.split('\n')] if out else []


def cacheIt(root, cacheDir, caches, rm, op):
    op = f"{op}S3" if cacheDir.startswith('s3://') else op
    for cache in caches:
        bkPath = cache.replace(root, cacheDir)
        if op == 'fromBk':
            if rm:
                print(f"deleting: {dirop(bkPath, rm=True, mkdir=False)}, {dirop(cache, rm=True, mkdir=False)}")
            if exists(bkPath) and not exists(cache):
                print(f"fromBk: {dirop(bkPath, symDir=dirname(cache))}")
        elif op == 'toBk':
            if rm:
                print(f"deleting: {dirop(bkPath, rm=True, mkdir=False)}, {dirop(cache, rm=True, mkdir=False)}")
            if not exists(bkPath) and exists(cache):
                print(f"toBk: {dirop(cache, cpDir=dirname(bkPath))}")
        elif op == 'fromBkS3':
            isDir = False if os.path.splitext(cache)[-1] else True
            if rm:
                s3rm(isDir, bkPath)
            if not exists(cache) and s3exists(bkPath):
                print(f"fromBkS3: {downloadS3(isDir, bkPath, dirname(cache), verbose=False)}")
        elif op == 'toBkS3':
            if exists(cache):
                isDir = False if os.path.splitext(cache)[-1] else True
                assert isDir == os.path.isdir(cache), f"{cache} is {'file' if isDir else 'dir'}, add/remove extension"
                if rm:
                    s3rm(isDir, bkPath)
                if not s3exists(bkPath):
                    print(f"toBkS3: {uploadS3(isDir, cache, dirname(bkPath), verbose=False)}")
