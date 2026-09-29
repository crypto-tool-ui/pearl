'use strict';

// Legitimate Pearl miner-to-proxy channel. This module has no persistence, shell execution,
// wallet handling, or process hooks. It only implements the documented authenticated channel.
const {
  createCipheriv,
  createDecipheriv,
  createPrivateKey,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  randomBytes,
} = require('node:crypto');

const MAGIC = Buffer.from('PRL7', 'ascii');
const VERSION = 1;
const CLIENT_HELLO = 1;
const SERVER_HELLO = 2;
const DATA_FRAME = 16;
const CLIENT_TO_PROXY = 1;
const PROXY_TO_CLIENT = 2;
const MAX_FRAME_BYTES = 1024 * 1024;
const MAX_COUNTER = 0xffff_ffff_ffff_fffen;
const SALT = Buffer.from('pearl-e2e-v1-salt', 'ascii');
const CLIENT_INFO = Buffer.from('pearl-e2e-v1 client-to-proxy', 'ascii');
const PROXY_INFO = Buffer.from('pearl-e2e-v1 proxy-to-client', 'ascii');
const AAD_PREFIX = Buffer.from('pearl-e2e-v1 aad', 'ascii');
const X25519_PRIVATE_PREFIX = Buffer.from('302e020100300506032b656e04220420', 'hex');
const X25519_PUBLIC_PREFIX = Buffer.from('302a300506032b656e032100', 'hex');

class CryptoError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'CryptoError';
    this.code = code;
  }
}

function bytes(value, length, name) {
  const result = Buffer.from(value);
  if (result.length !== length) throw new CryptoError('invalid_length', `${name} must be ${length} bytes`);
  return result;
}

function privateKeyFromRaw(raw) {
  return createPrivateKey({
    key: Buffer.concat([X25519_PRIVATE_PREFIX, bytes(raw, 32, 'private key')]),
    format: 'der',
    type: 'pkcs8',
  });
}

function publicKeyFromRaw(raw) {
  return createPublicKey({
    key: Buffer.concat([X25519_PUBLIC_PREFIX, bytes(raw, 32, 'public key')]),
    format: 'der',
    type: 'spki',
  });
}

function rawPrivateKey(key) {
  const der = key.export({ format: 'der', type: 'pkcs8' });
  return Buffer.from(der).subarray(-32);
}

function rawPublicKey(key) {
  const der = key.export({ format: 'der', type: 'spki' });
  return Buffer.from(der).subarray(-32);
}

function generateX25519KeyPair() {
  const { privateKey, publicKey } = generateKeyPairSync('x25519');
  return { privateKey: rawPrivateKey(privateKey), publicKey: rawPublicKey(publicKey) };
}

function frame(body) {
  if (body.length > MAX_FRAME_BYTES + 64) throw new CryptoError('frame_too_large', 'frame exceeds the maximum size');
  const prefix = Buffer.alloc(4);
  prefix.writeUInt32BE(body.length, 0);
  return Buffer.concat([prefix, body]);
}

function parseFrame(encoded) {
  const input = Buffer.from(encoded);
  if (input.length < 4) throw new CryptoError('truncated', 'frame is truncated');
  const declared = input.readUInt32BE(0);
  if (declared > MAX_FRAME_BYTES + 64) throw new CryptoError('frame_too_large', 'frame exceeds the maximum size');
  if (declared !== input.length - 4) throw new CryptoError('invalid_length', 'frame length is invalid');
  return input.subarray(4);
}

function checkHeader(body, type) {
  if (body.length < 6 || !body.subarray(0, 4).equals(MAGIC) || body[4] !== VERSION) {
    throw new CryptoError('invalid_header', 'unknown protocol magic or version');
  }
  if (body[5] !== type) throw new CryptoError('unexpected_handshake', 'unexpected message type');
}

function derive(role, sessionId, ephemeralShared, identityShared) {
  const ikm = Buffer.concat([ephemeralShared, identityShared]);
  const clientInfo = Buffer.concat([CLIENT_INFO, sessionId]);
  const proxyInfo = Buffer.concat([PROXY_INFO, sessionId]);
  const clientKey = Buffer.from(hkdfSync('sha256', ikm, SALT, clientInfo, 32));
  const proxyKey = Buffer.from(hkdfSync('sha256', ikm, SALT, proxyInfo, 32));
  return new Session(
    role,
    sessionId,
    role === 'client' ? clientKey : proxyKey,
    role === 'client' ? proxyKey : clientKey,
  );
}

function associatedData(sessionId, direction, counter) {
  const counterBytes = Buffer.alloc(8);
  counterBytes.writeBigUInt64BE(BigInt(counter));
  return Buffer.concat([AAD_PREFIX, sessionId, Buffer.from([direction]), counterBytes]);
}

function nonce(direction, counter) {
  const value = Buffer.alloc(12);
  value[0] = direction;
  value.writeBigUInt64BE(BigInt(counter), 1);
  return value;
}

class Session {
  constructor(role, sessionId, sendKey, recvKey) {
    this.role = role;
    this.sessionId = bytes(sessionId, 16, 'session id');
    this.sendKey = bytes(sendKey, 32, 'send key');
    this.recvKey = bytes(recvKey, 32, 'receive key');
    this.sendCounter = 0n;
    this.recvCounter = 0n;
  }

  encrypt(plaintext) {
    const message = Buffer.from(plaintext);
    if (message.length > MAX_FRAME_BYTES) throw new CryptoError('frame_too_large', 'payload exceeds the maximum size');
    if (this.sendCounter >= MAX_COUNTER) throw new CryptoError('rekey_required', 'new handshake required');
    const direction = this.role === 'client' ? CLIENT_TO_PROXY : PROXY_TO_CLIENT;
    const counter = this.sendCounter;
    const cipher = createCipheriv('aes-256-gcm', this.sendKey, nonce(direction, counter));
    cipher.setAAD(associatedData(this.sessionId, direction, counter));
    const ciphertext = Buffer.concat([cipher.update(message), cipher.final(), cipher.getAuthTag()]);
    const body = Buffer.concat([
      MAGIC,
      Buffer.from([VERSION, DATA_FRAME, direction]),
      (() => { const value = Buffer.alloc(8); value.writeBigUInt64BE(counter); return value; })(),
      ciphertext,
    ]);
    this.sendCounter += 1n;
    return frame(body);
  }

  decrypt(encoded) {
    const body = parseFrame(encoded);
    if (body.length < 4 + 1 + 1 + 1 + 8 + 16) throw new CryptoError('truncated', 'encrypted frame is truncated');
    checkHeader(body, DATA_FRAME);
    const expectedDirection = this.role === 'client' ? PROXY_TO_CLIENT : CLIENT_TO_PROXY;
    if (body[6] !== expectedDirection) throw new CryptoError('invalid_direction', 'frame direction is invalid');
    const counter = body.readBigUInt64BE(7);
    if (counter >= MAX_COUNTER) throw new CryptoError('rekey_required', 'new handshake required');
    if (counter !== this.recvCounter) {
      throw new CryptoError('counter_mismatch', `expected counter ${this.recvCounter}, got ${counter}`);
    }
    try {
      const decipher = createDecipheriv('aes-256-gcm', this.recvKey, nonce(expectedDirection, counter));
      decipher.setAAD(associatedData(this.sessionId, expectedDirection, counter));
      decipher.setAuthTag(body.subarray(body.length - 16));
      const plaintext = Buffer.concat([decipher.update(body.subarray(15, body.length - 16)), decipher.final()]);
      this.recvCounter += 1n;
      return plaintext;
    } catch (error) {
      throw new CryptoError('authentication_failed', 'AES-GCM authentication failed');
    }
  }
}

function clientHello({ sessionId = randomBytes(16), ephemeralPrivate = generateX25519KeyPair().privateKey, pinnedServerStatic = null } = {}) {
  const session = bytes(sessionId, 16, 'session id');
  const privateKey = bytes(ephemeralPrivate, 32, 'ephemeral private key');
  const publicKey = rawPublicKey(createPublicKey(privateKeyFromRaw(privateKey)));
  const body = Buffer.concat([MAGIC, Buffer.from([VERSION, CLIENT_HELLO]), session, publicKey]);
  return { frame: frame(body), sessionId: session, ephemeralPrivate: privateKey, ephemeralPublic: publicKey, pinnedServerStatic };
}

function serverHello(client, { identityPrivate, ephemeralPrivate = generateX25519KeyPair().privateKey } = {}) {
  const body = parseFrame(client.frame || client);
  if (body.length !== 4 + 1 + 1 + 16 + 32) throw new CryptoError('invalid_length', 'client hello length is invalid');
  checkHeader(body, CLIENT_HELLO);
  const sessionId = body.subarray(6, 22);
  const clientPublic = publicKeyFromRaw(body.subarray(22, 54));
  const identity = bytes(identityPrivate, 32, 'identity private key');
  const ephemeral = bytes(ephemeralPrivate, 32, 'ephemeral private key');
  const serverEphemeralPublic = rawPublicKey(createPublicKey(privateKeyFromRaw(ephemeral)));
  const serverStaticPublic = rawPublicKey(createPublicKey(privateKeyFromRaw(identity)));
  const ee = diffieHellman({ privateKey: privateKeyFromRaw(ephemeral), publicKey: clientPublic });
  const es = diffieHellman({ privateKey: privateKeyFromRaw(identity), publicKey: clientPublic });
  const session = derive('proxy', sessionId, ee, es);
  const response = frame(Buffer.concat([MAGIC, Buffer.from([VERSION, SERVER_HELLO]), sessionId, serverEphemeralPublic, serverStaticPublic]));
  return { frame: response, session, serverStaticPublic, ephemeralPrivate: ephemeral };
}

function finishClient(client, serverFrame) {
  const body = parseFrame(serverFrame);
  if (body.length !== 4 + 1 + 1 + 16 + 32 + 32) throw new CryptoError('invalid_length', 'server hello length is invalid');
  checkHeader(body, SERVER_HELLO);
  if (!body.subarray(6, 22).equals(client.sessionId)) throw new CryptoError('session_mismatch', 'session id does not match');
  const serverEphemeral = publicKeyFromRaw(body.subarray(22, 54));
  const serverStatic = body.subarray(54, 86);
  if (client.pinnedServerStatic && !bytes(client.pinnedServerStatic, 32, 'pinned server key').equals(serverStatic)) {
    throw new CryptoError('pinned_key_mismatch', 'server identity key does not match the pin');
  }
  const ee = diffieHellman({ privateKey: privateKeyFromRaw(client.ephemeralPrivate), publicKey: serverEphemeral });
  const es = diffieHellman({ privateKey: privateKeyFromRaw(client.ephemeralPrivate), publicKey: publicKeyFromRaw(serverStatic) });
  return derive('client', client.sessionId, ee, es);
}

module.exports = {
  AAD_PREFIX,
  CLIENT_TO_PROXY,
  MAX_COUNTER,
  MAX_FRAME_BYTES,
  PROXY_TO_CLIENT,
  CryptoError,
  Session,
  clientHello,
  finishClient,
  generateX25519KeyPair,
  serverHello,
};
