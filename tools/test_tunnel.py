#!/usr/bin/env python3
"""Run the Rust transport through the real tunnel and Node proxy against an echo backend."""
import asyncio
import os
import json
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def ready(port, proc):
    for _ in range(100):
        if proc.returncode is not None:
            raise RuntimeError('proxy/tunnel exited before readiness')
        try:
            _, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(.05)
    raise TimeoutError('proxy/tunnel did not listen')


async def main():
    async def echo(reader, writer):
        try:
            while data := await reader.readline():
                if data == b'close\n':
                    break
                try:
                    request = json.loads(data)
                except ValueError:
                    request = {}
                if isinstance(request, dict) and request.get('method') in ('mining.subscribe', 'mining.authorize'):
                    writer.write((json.dumps({'id': request['id'], 'result': True, 'error': None}) + '\n').encode())
                    if request['method'] == 'mining.authorize':
                        job = {'method': 'mining.notify', 'params': {'job_id': 'integration_1',
                               'header': '00' * 76, 'target': 'ff' * 32, 'cert_version': 3}}
                        writer.write((json.dumps(job) + '\n').encode())
                else:
                    writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    backend = await asyncio.start_server(echo, '127.0.0.1', 0, limit=4*1024*1024)
    backend_port = backend.sockets[0].getsockname()[1]
    ws_port, tunnel_port = free_port(), free_port()
    env = dict(os.environ, WS_HOST='127.0.0.1', WS_PORT=str(ws_port),
               TCP_TARGET_HOST='127.0.0.1', TCP_TARGET_PORT=str(backend_port),
               WS_URL=f'ws://127.0.0.1:{ws_port}', LISTEN_HOST='127.0.0.1',
               LISTEN_PORT=str(tunnel_port), DISABLE_GPU_STATS='1',
               PEARL_TEST_TUNNEL=f'127.0.0.1:{tunnel_port}')
    children = []
    with tempfile.TemporaryFile() as logs:
        try:
            for command, port in [(['node', 'proxy/src/server.js'], ws_port),
                                  ([sys.executable, 'tunel.py'], tunnel_port)]:
                proc = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
                                                           stdout=logs, stderr=logs)
                children.append(proc)
                await ready(port, proc)
            test = await asyncio.create_subprocess_exec(
                'cargo', 'test', '--locked', '-p', 'pearl-net', '--lib',
                'real_proxy_tunnel', '--', '--ignored', '--nocapture', cwd=ROOT, env=env)
            children.append(test)
            code = await asyncio.wait_for(test.wait(), 180)
            if code:
                logs.seek(0)
                print(logs.read().decode(errors='replace'))
                raise RuntimeError(f'Rust integration test failed: {code}')
        finally:
            for proc in reversed(children):
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), 3)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
            backend.close()
            await backend.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())
