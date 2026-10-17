"""ROS camera subprocess bridge for the non-ROS inference environment."""
from __future__ import annotations
import io
import os
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class RosHeadCamera:
    def __init__(self,args,with_depth=False):
        from .head_camera import HeadCamera
        # Validate the same identity and stream overrides before starting ROS.
        HeadCamera(args,with_depth)
        self.directory=args.calibration_dir
        self.with_depth=with_depth
        self.process=None
        self.frames=deque(maxlen=1)
        self.errors=deque(maxlen=8)
        self.condition=threading.Condition()
        self.threads=[]

    def __enter__(self):
        script=Path(__file__).with_name('ros_head_stream.py')
        command=['bash','-c','source /opt/ros/humble/setup.bash; exec /usr/bin/python3 "$@"',
                 'head-face-reader',str(script),'--calibration-dir',str(self.directory)]
        if self.with_depth:command.append('--with-depth')
        env=os.environ.copy()
        env.setdefault('ROS_LOCALHOST_ONLY','1')
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        self.threads=[threading.Thread(target=self._reader,daemon=True),threading.Thread(target=self._errors,daemon=True)]
        for t in self.threads:t.start()
        try:
            self.first=self._next()
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def _errors(self):
        for line in self.process.stderr:
            with self.condition:self.errors.append(line.decode(errors='replace').strip())

    def _reader(self):
        def exact(size):
            data=bytearray()
            while len(data)<size:
                part=self.process.stdout.read(size-len(data))
                if not part:raise EOFError('ROS head image stream closed')
                data.extend(part)
            return data
        try:
            while True:
                size=struct.unpack('!I',exact(4))[0]
                if not 0<size<64*1024*1024:raise ValueError('Invalid ROS image packet size')
                with np.load(io.BytesIO(exact(size)),allow_pickle=False) as f:
                    color=f['color'];depth=f['depth'] if self.with_depth else None
                with self.condition:
                    self.frames.append((time.monotonic(),color,depth));self.condition.notify_all()
        except Exception as exc:
            with self.condition:self.errors.append(str(exc));self.condition.notify_all()

    def _next(self):
        deadline=time.monotonic()+10
        with self.condition:
            while time.monotonic()<deadline:
                if self.frames:
                    t,color,depth=self.frames.pop()
                    if time.monotonic()-t<=.5:return (color,depth) if self.with_depth else color
                if self.process.poll() is not None:break
                self.condition.wait(timeout=.1)
            raise RuntimeError('Head ROS camera unavailable: '+'; '.join(self.errors))

    def read(self):
        if getattr(self,'first',None) is not None:
            # Fetch the latest packet; startup verification frames may be old after model initialization.
            self.first=None
        return self._next()

    def __exit__(self,exc_type,exc_value,traceback):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            for t in self.threads:t.join(timeout=1)
            self.process.stdout.close();self.process.stderr.close()
            self.process=None
