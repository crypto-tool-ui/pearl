'use strict';

const assert = require('node:assert/strict');
const net = require('node:net');
const test = require('node:test');
const channel = require('./crypto/index.js');
const { startProxy } = require('./server.js');

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function nextFrame(socket) {
  return new Promise((resolve, reject) => {
    let pending = Buffer.alloc(0);
    const onData = (chunk) => {
      pending = Buffer.concat([pending, chunk]);
      if (pending.length < 4) return;
      const length = pending.readUInt32BE(0);
      if (pending.length < length + 4) return;
      socket.off('data', onData);
      resolve(pending.subarray(0, length + 4));
    };
    socket.on('data', onData);
    socket.once('error', reject);
  });
}

test('encrypted miner payload is relayed to Kryptex and response is encrypted back', async () => {
  let forwarded;
  const pool = net.createServer((socket) => {
    socket.on('data', (data) => {
      forwarded = Buffer.from(data);
      socket.write('{"id":1,"result":true}\n');
    });
  });
  const poolPort = await listen(pool);
  const proxy = startProxy({
    host: '127.0.0.1',
    port: 0,
    endpoint: { tls: false, host: '127.0.0.1', port: poolPort },
    identityPrivate: Buffer.alloc(32, 0x33),
  });
  await new Promise((resolve) => proxy.once('listening', resolve));
  const proxyPort = proxy.address().port;
  const miner = net.connect(proxyPort, '127.0.0.1');

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
    proxy.close();
    pool.close();
  }
});
