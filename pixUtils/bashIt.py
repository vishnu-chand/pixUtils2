import os
import sys
import json
import time
import traceback
import subprocess as sp
from os.path import basename
from datetime import datetime as dt

# bashEnv = os.getenv('bashEnv', 'export PATH=/home/ec2-user/miniconda3/bin:$PATH;eval "$(conda shell.bash hook)";conda activate gputf;')


bashEnv = os.getenv('bashEnv', f'export PATH="{os.path.dirname(sys.executable)}:$PATH";')


def decodeCmd(cmd, sepBy):
    cmd = [cmd.strip() for cmd in cmd.split('\n')]
    cmd = [cmd for cmd in cmd if cmd and not cmd.startswith('#')]
    cmd = sepBy.join(cmd)
    return cmd


def decodeCmd2(cmds):
    cmds = [cmd.strip() for cmd in cmds]
    cmd = [cmd for cmd in cmds if cmd and not cmd.startswith('#')]
    return cmd


def getTraceBack(searchPys, tracebackData=None):
    errorTraceBooks = [basename(p) for p in searchPys]
    oTrace = tracebackData or traceback.format_exc()
    trace = oTrace.strip().split('\n')
    msg = trace[-1]
    flow = ''
    for oLine in trace:
        line = oLine.strip()
        if line.startswith('File "'):
            line = line[6:].split('", line')[0]
            if not errorTraceBooks or basename(line) in errorTraceBooks:
                flow += f"\n{oLine}"
    traces = f"""
{oTrace}

{flow}
"""
    return msg, traces


def exeIt(cmd, returnOutput=True, waitTillComplete=True, sepBy=' ', inData=None, debug=True, raiseOnException=True, skipExe=False, dispCmd=True):
    if returnOutput and not waitTillComplete:
        raise Exception("waitTillComplete is False, to get returnOutput set waitTillComplete to True")
    stdin = None if inData is None else sp.PIPE
    stdout, stderr = (None, None) if debug else (sp.DEVNULL, sp.DEVNULL)
    if returnOutput:
        stdout, stderr = sp.PIPE, sp.PIPE
    cmd = f"{bashEnv}{decodeCmd(cmd, sepBy)}".rstrip(';')
    __cmd = cmd.replace(';', ';\n\t')
    errCode, out, err = 0, 'no output', 'no output'
    if dispCmd and debug and not skipExe:
        print(f"""

    {__cmd}
    """)
    if skipExe:
        print(f"""
        skipping execution of:

    {__cmd}
    """)
    elif not returnOutput and stdin is None:
        if waitTillComplete:
            errCode = os.system(cmd)
        else:
            errCode = os.system(f"{cmd} &")
    else:
        p1 = sp.Popen(cmd, shell=True, stdin=stdin, stdout=stdout, stderr=stderr)
        errCode, out, err = 0, '', ''
        if waitTillComplete:
            inData = None if inData is None else inData.encode()
            out, err = p1.communicate(inData)
            errCode = p1.poll()
            out = '' if out is None else out.decode().strip()
            err = '' if err is None else err.decode().strip()
    if raiseOnException and errCode:
        tracebackData = f"""
        cmd         : {__cmd if dispCmd else 'cmd is hidden; dispCmd == False'}
        returnCode  : {errCode}
        out1[out]   : {out}
        out2[err]   : {err}
        """
        if debug:
            tracebackData += f"""
        stderr      : {stderr}
        stdin       : {stdin}
        stdout      : {stdout}
        """
        if returnOutput:
            _, tracebackData = getTraceBack([], tracebackData=tracebackData)
        raise Exception(f"subprocess failed: {tracebackData}")
    if debug and not skipExe:
        print(f"""
      _____________________________________________________________________________________
      cmd         : {__cmd if dispCmd else 'cmd is hidden; dispCmd == False'}
      returnCode  : {errCode}
      out1[out]   : {out}
      out2[err]   : {err}
      stderr      : {stderr}
      stdin       : {stdin}
      stdout      : {stdout}

      _____________________________________________________________________________________
      """)
    return __cmd if dispCmd else 'cmd is hidden; dispCmd == False', errCode, out, err


def curlIt(url, data=None, method='POST', other='', timeout=60, debug=False, waitTillComplete=False, skipExe=False):
    timeout = f'--max-time {timeout}' if timeout else ''
    curlCmd = f"curl -X {method.upper()} '{url}' {timeout} {other}"
    jsonPath = None
    if data is not None:
        jsonPath = f'/tmp/{dt.now().strftime("%b%d_%H_%M_%S%f")}.json'
        with open(jsonPath, 'w') as book:
            json.dump(data, book)
        curlCmd = f"curl -X {method.upper()} --data '@{jsonPath}' '{url}' {timeout} {other};"
    res = exeIt(cmd=curlCmd, returnOutput=waitTillComplete, waitTillComplete=waitTillComplete, sepBy='', debug=debug, skipExe=skipExe)
    time.sleep(.1)
    if jsonPath is not None:
        os.remove(jsonPath)
    return res
