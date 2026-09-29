'use strict';

const assert = require('node:assert/strict');
const net = require('node:net');
const test = require('node:test');
const channel = require('./crypto/index.js');
const { connectPool, startProxy } = require('./server.js');

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function nextFrame(socket) {
  return new Promise((resolve, reject) => {
    let pending = Buffer.alloc(0);
    const cleanup = () => {
      clearTimeout(timer);
      socket.off('data', onData);
      socket.off('error', onError);
      socket.off('close', onClose);
    };
    const fail = (error) => { cleanup(); reject(error); };
    const onError = (error) => fail(error);
    const onClose = () => fail(new Error('socket closed before a complete frame'));
    const onData = (chunk) => {
      pending = Buffer.concat([pending, chunk]);
      if (pending.length < 4) return;
      const length = pending.readUInt32BE(0);
      if (pending.length < length + 4) return;
      cleanup();
      resolve(pending.subarray(0, length + 4));
    };
    const timer = setTimeout(() => fail(new Error('timed out waiting for proxy frame')), 3000);
    socket.on('data', onData);
    socket.once('error', onError);
    socket.once('close', onClose);
  });
}

// server.close() stops listening but does not destroy existing connections.
function trackServer(t, server) {
  const sockets = new Set();
  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.once('close', () => sockets.delete(socket));
  });
  t.after(() => new Promise((resolve, reject) => {
    for (const socket of sockets) socket.destroy();
    server.close((error) => {
      if (error && error.code !== 'ERR_SERVER_NOT_RUNNING') reject(error);
      else resolve();
    });
  }));
  return server;
}

test('encrypted miner payload is relayed to Kryptex and response is encrypted back', { timeout: 10000 }, async (t) => {
  let forwarded;
  const pool = net.createServer((socket) => {
    socket.on('data', (data) => {
      forwarded = Buffer.from(data);
      socket.write('{"id":1,"result":true}\n');
    });
  });
  trackServer(t, pool);
  const poolPort = await listen(pool);
  const proxy = startProxy({
    host: '127.0.0.1',
    port: 0,
    endpoint: { tls: false, host: '127.0.0.1', port: poolPort },
    identityPrivate: Buffer.alloc(32, 0x33),
  });
  trackServer(t, proxy);
  await new Promise((resolve, reject) => {
    proxy.once('listening', resolve);
    proxy.once('error', reject);
  });
  const proxyPort = proxy.address().port;
  const miner = net.connect(proxyPort, '127.0.0.1');
  t.after(() => miner.destroy());

  try {
    const clientHello = channel.clientHello({
      sessionId: Buffer.alloc(16, 0x44),
      ephemeralPrivate: Buffer.alloc(32, 0x11),
    });
    miner.write(clientHello.frame);
    const serverFrame = await nextFrame(miner);
    const clientSession = channel.finishClient(clientHello, serverFrame);
    const payload = Buffer.from('{"id":1,"method":"mining.subscribe","params":[]}\n');
    miner.write(clientSession.encrypt(payload));
    const encryptedResponse = await nextFrame(miner);
    assert.deepEqual(clientSession.decrypt(encryptedResponse), Buffer.from('{"id":1,"result":true}\n'));
    assert.deepEqual(forwarded, payload);
  } finally {
    miner.destroy();
  }
});

test('connectPool returns the connected socket', { timeout: 5000 }, async (t) => {
  const pool = trackServer(t, net.createServer());
  const port = await listen(pool);
  const socket = await connectPool({ tls: false, host: '127.0.0.1', port });
  t.after(() => socket.destroy());
  assert.ok(socket instanceof net.Socket);
  assert.equal(socket.remotePort, port);
  assert.equal(socket.connecting, false);
});

test('frame reader rejects a peer closing before its response', { timeout: 5000 }, async (t) => {
  const pool = trackServer(t, net.createServer((socket) => socket.end()));
  const port = await listen(pool);
  const socket = net.connect(port, '127.0.0.1');
  t.after(() => socket.destroy());
  await assert.rejects(nextFrame(socket), /closed before a complete frame/);
});
