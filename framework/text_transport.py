"""Single-flight simulator pipes with deadlines covering writes and replies."""
from __future__ import annotations

import queue
import threading
import time


class TransportCancelled(RuntimeError):
    pass


class TextTransport:
    def __init__(self, process, responses, *, cancel=None, timeout=30):
        self.process, self.responses = process, responses
        self.cancelled = cancel.is_set if cancel is not None else lambda: False
        self.timeout = timeout
        self.closed = threading.Event()
        self._writes = queue.Queue(maxsize=1)
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._readers = []
        self._writer = threading.Thread(target=self._write, daemon=True)
        self._writer.start()

    def start_readers(self, *targets):
        for target in targets:
            thread = threading.Thread(target=target, daemon=True)
            self._readers.append(thread)
            thread.start()

    def emit(self, value):
        while not self.closed.is_set():
            try:
                self.responses.put(value, timeout=.05)
                return
            except queue.Full:
                continue

    def _write(self):
        while True:
            message = self._writes.get()
            if message is None or self.closed.is_set():
                return
            try:
                self.process.stdin.write(message)
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                self.emit(error)
                return

    def exchange(self, message):
        return self.exchange_many([message])[0]

    def exchange_many(self,messages):
        """Pipeline existing ordered commands through one owned writer.

        Replies are drained while the writer runs. Each received reply renews
        the existing per-command fault timeout, including a blocked write.
        """
        messages=list(messages)
        if not messages:return[]
        message='\n'.join(messages)
        with self._request_lock:
            deadline = time.monotonic() + self.timeout
            try:
                if self.closed.is_set():
                    raise OSError("Simulator transport is closed")
                if self.cancelled():
                    raise TransportCancelled("Simulator request cancelled")
                self._writes.put_nowait(message + "\n")
                values=[]
                while len(values)<len(messages):
                    if self.cancelled():
                        raise TransportCancelled("Simulator request cancelled")
                    if self.closed.is_set():
                        raise OSError("Simulator transport closed during request")
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Simulator request exceeded {self.timeout:g}s")
                    try:
                        value = self.responses.get(timeout=min(.05, remaining))
                    except queue.Empty:
                        continue
                    if isinstance(value, Exception):
                        raise value
                    if value is None:
                        raise OSError(f"Simulator stopped (exit={self.process.poll()})")
                    values.append(value)
                    if len(values)<len(messages):deadline=time.monotonic()+self.timeout
                return values
            except BaseException as error:
                first=getattr(self,'failure',None)
                if first is not None:raise first
                self.failure=error
                if isinstance(error,TimeoutError):error.request_message=message
                try:
                    self.close()
                except Exception as cleanup:
                    note=getattr(error,'add_note',None)
                    if note:note('Simulator cleanup: '+str(cleanup))
                raise

    def close(self):
        # Kill before touching TextIOWrapper: its write/read locks can be held
        # by blocked pipe threads, and closing the stream first would wait.
        self.closed.set()
        with self._close_lock:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            try:
                self._writes.put_nowait(None)
            except queue.Full:
                pass
            for thread in (self._writer, *self._readers):
                if thread is not threading.current_thread():
                    thread.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()
