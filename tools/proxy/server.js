'use strict';

// Pearl encrypted-channel terminator and Kryptex Stratum relay.
// The proxy never interprets or rewrites Stratum payloads: it only decrypts the miner frame,
// forwards the bytes to the configured pool connection, and encrypts each newline-delimited pool
// response back to the miner.
const net = require('node:net');
const tls = require('node:tls');
const { randomBytes } = require('node:crypto');
const { serverHello } = require('./crypto/index.js');

const MAX_FRAME_BYTES = 1024 * 1024;
const MAX_WIRE_BODY = MAX_FRAME_BYTES + 64;
const MAX_POOL_BUFFER = 4 * 1024 * 1024;

function requiredIdentity() {
  const value = process.env.PEARL_PROXY_IDENTITY_PRIVATE_HEX;
  if (!value || !/^[0-9a-fA-F]{64}$/.test(value)) {
    throw new Error('PEARL_PROXY_IDENTITY_PRIVATE_HEX must be a 32-byte hex key');
  }
  return Buffer.from(value, 'hex');
}

function parsePoolEndpoint(value) {
  if (!value) {
    throw new Error('PEARL_POOL_ENDPOINT is required, for example stratum+ssl://host:8048');
  }
  const match = /^(stratum\+ssl|stratum\+tcp):\/\/([^:]+):(\d+)$/.exec(value);
  if (!match) {
    throw new Error('PEARL_POOL_ENDPOINT must be stratum+ssl://host:port or stratum+tcp://host:port');
  }
  const port = Number(match[3]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('PEARL_POOL_ENDPOINT port is invalid');
  }
  return { tls: match[1] === 'stratum+ssl', host: match[2], port };
}

function connectPool(endpoint) {
  return new Promise((resolve, reject) => {
    const onConnect = (socket) => {
      socket.setNoDelay(true);
      resolve(socket);
    };
    const socket = endpoint.tls
      ? tls.connect({ host: endpoint.host, port: endpoint.port, servername: endpoint.host }, onConnect)
      : net.connect({ host: endpoint.host, port: endpoint.port }, onConnect);
    socket.once('error', reject);
  });
}

function takeFrames(buffer) {
  const frames = [];
  let pending = buffer;
  while (pending.length >= 4) {
    const bodyLength = pending.readUInt32BE(0);
    if (bodyLength > MAX_WIRE_BODY) throw new Error('encrypted frame exceeds the maximum size');
    if (pending.length < bodyLength + 4) break;
    frames.push(pending.subarray(0, bodyLength + 4));
    pending = pending.subarray(bodyLength + 4);
  }
  if (pending.length > MAX_WIRE_BODY + 4) throw new Error('incomplete encrypted frame exceeds the maximum size');
  return { frames, pending };
}

function relayConnection(miner, endpoint, identityPrivate) {
  let encryptedPending = Buffer.alloc(0);
  let poolPending = Buffer.alloc(0);
  let session = null;
  let pool = null;
  let closed = false;
  let work = Promise.resolve();

  const close = () => {
    if (closed) return;
    closed = true;
    if (pool) pool.destroy();
    miner.destroy();
  };

  const attachPool = (candidate) => {
    pool = candidate;
    pool.on('data', (chunk) => {
      if (closed) return;
      poolPending = Buffer.concat([poolPending, chunk]);
      if (poolPending.length > MAX_POOL_BUFFER) return close();
      while (true) {
        const newline = poolPending.indexOf(0x0a);
        if (newline < 0) return;
        const line = poolPending.subarray(0, newline + 1);
        poolPending = poolPending.subarray(newline + 1);
        if (line.length > MAX_FRAME_BYTES) return close();
        try {
          miner.write(session.encrypt(line));
        } catch {
          return close();
        }
      }
    });
    pool.once('error', close);
    pool.once('close', close);
  };

  const processFrames = async () => {
    const extracted = takeFrames(encryptedPending);
    encryptedPending = extracted.pending;
    for (const frame of extracted.frames) {
      if (session === null) {
        const hello = serverHello(frame, { identityPrivate, ephemeralPrivate: randomBytes(32) });
        session = hello.session;
        miner.write(hello.frame);
        attachPool(await connectPool(endpoint));
      } else {
        const plaintext = session.decrypt(frame);
        if (!pool || plaintext.length === 0) throw new Error('pool connection is not ready');
        pool.write(plaintext);
      }
    }
  };

  miner.setNoDelay(true);
  miner.on('data', (chunk) => {
    if (closed) return;
    encryptedPending = Buffer.concat([encryptedPending, chunk]);
    if (encryptedPending.length > MAX_POOL_BUFFER) return close();
    work = work.then(processFrames).catch(close);
  });
  miner.once('error', close);
  miner.once('close', close);
}

function startProxy({ host = process.env.PEARL_PROXY_LISTEN_HOST || '127.0.0.1', port = Number(process.env.PEARL_PROXY_PORT || 9836), endpoint = parsePoolEndpoint(process.env.PEARL_POOL_ENDPOINT), identityPrivate = requiredIdentity() } = {}) {
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('PEARL_PROXY_PORT is invalid');
  const server = net.createServer((socket) => relayConnection(socket, endpoint, identityPrivate));
  server.on('error', (error) => {
    console.error(`[pearl-proxy] server error: ${error.message}`);
  });
  server.listen(port, host, () => {
    const address = server.address();
    console.log(`[pearl-proxy] listening on ${typeof address === 'string' ? address : `${address.address}:${address.port}`}`);
    console.log(`[pearl-proxy] pool ${endpoint.tls ? 'TLS' : 'TCP'} ${endpoint.host}:${endpoint.port}`);
  });
  return server;
}

if (require.main === module) startProxy();

module.exports = { connectPool, parsePoolEndpoint, startProxy, takeFrames };
