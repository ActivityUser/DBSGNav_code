"""终端输出同时写入文件，便于留存完整日志做分析。"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


class Tee(TextIO):
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log = log_file

    def fileno(self) -> int:
        """forkserver/spawn 会读取 stdout/stderr 的 fd；必须落到真实终端 fd，不能缺省。"""
        return self._stream.fileno()

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._log.write(s)
        self._stream.flush()
        self._log.flush()
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextmanager
def tee_session_log(path: Path, *, encoding: str = "utf-8"):
    """将 sys.stdout / sys.stderr 同时镜像到 path（上下文结束后恢复）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_f = path.open("w", encoding=encoding)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    try:
        yield path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()
