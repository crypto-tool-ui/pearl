/**
 * E2E Encryption — mirrors src/e2e_crypto.h in the miner.
 * Uses Node.js native X25519 via JWK format.
 */

const crypto = require('crypto');
const HKDF_SALT = Buffer.from('csd-pool-e2e-v1');
const HKDF_INFO_C2S = Buffer.from('aes-256-gcm-key-c2s');
const HKDF_INFO_S2C = Buffer.from('aes-256-gcm-key-s2c');

/** Import raw 32-byte public key as a KeyObject (JWK format) */
function pubKeyObj(raw) {
  return crypto.createPublicKey({
    key: { kty: 'OKP', crv: 'X25519', x: raw.toString('base64url') },
    format: 'jwk',
  });
}

function generateKeypair(privateHex) {
  let kp;
  if (privateHex) {
    if (!/^[0-9a-f]{64}$/i.test(privateHex)) throw new Error('PROXY_PRIVATE_KEY must be 32-byte hex');
    const d = Buffer.from(privateHex, 'hex');
    // RFC 8410 PKCS#8 wrapper for a raw X25519 scalar (lets Node derive x).
    const der = Buffer.concat([Buffer.from('302e020100300506032b656e04220420', 'hex'), d]);
    const priv = crypto.createPrivateKey({ key: der, format: 'der', type: 'pkcs8' });
    kp = { privateKey: priv, publicKey: crypto.createPublicKey(priv) };
  } else {
    kp = crypto.generateKeyPairSync('x25519');
  }
  return {
    privateKeyObj: kp.privateKey,
    publicKey: kp.publicKey.export({ type: 'spki', format: 'der' }).subarray(-32),
  };
}

function ecdh(privateKeyObj, peerPublicKeyRaw) {
  return crypto.diffieHellman({ privateKey: privateKeyObj, publicKey: pubKeyObj(peerPublicKeyRaw) });
}

function derivePublicKey(privateKeyObj) {
  return crypto.createPublicKey(privateKeyObj)
    .export({ type: 'spki', format: 'der' }).subarray(-32);
}

function deriveKey(sharedSecret, info) {
  return crypto.hkdfSync('sha256', sharedSecret, HKDF_SALT, info, 32);
}

function buildNonce(counter) {
  const nonce = Buffer.alloc(12);
  nonce.write('CSD!', 0, 'ascii');
  nonce.writeBigUInt64BE(BigInt(counter), 4);
  return nonce;
}

function createSession(sharedSecret, role = 'server') {
  if (role !== 'server' && role !== 'client') throw new Error('invalid E2E session role');
  const txKey = deriveKey(sharedSecret, role === 'server' ? HKDF_INFO_S2C : HKDF_INFO_C2S);
  const rxKey = deriveKey(sharedSecret, role === 'server' ? HKDF_INFO_C2S : HKDF_INFO_S2C);
  let txCounter = 0;
  let lastRxCounter = 0;
  return {
    encrypt(plaintext) {
      txCounter++;
      const nonce = buildNonce(txCounter);
      const cipher = crypto.createCipheriv('aes-256-gcm', txKey, nonce, { authTagLength: 16 });
      const enc = Buffer.concat([cipher.update(plaintext), cipher.final()]);
      const tag = cipher.getAuthTag();
      const wire = Buffer.alloc(1 + nonce.length + enc.length + tag.length);
      wire[0] = nonce.length;
      nonce.copy(wire, 1);
      enc.copy(wire, 1 + nonce.length);
      tag.copy(wire, 1 + nonce.length + enc.length);
      return wire;
    },
    decrypt(wire) {
      if (wire.length < 1 + 12 + 16 || wire[0] !== 12) throw new Error('E2E decrypt: invalid frame');
      const nLen = wire[0];
      if (wire.length < 1 + nLen + 16) throw new Error('E2E decrypt: invalid frame');
      const nonce = wire.subarray(1, 1 + nLen);
      if (nonce.subarray(0, 4).toString('ascii') !== 'CSD!') throw new Error('E2E decrypt: invalid nonce');
      let counter = 0n;
      for (const byte of nonce.subarray(4)) counter = (counter << 8n) | BigInt(byte);
      if (counter === 0n || counter <= lastRxCounter) throw new Error('E2E decrypt: replayed frame');
      const ct = wire.subarray(1 + nLen);
      const d = crypto.createDecipheriv('aes-256-gcm', rxKey, nonce, { authTagLength: 16 });
      d.setAuthTag(ct.subarray(ct.length - 16));
      const plain = Buffer.concat([d.update(ct.subarray(0, ct.length - 16)), d.final()]);
      lastRxCounter = counter;
      return plain;
    },
  };
}

module.exports = { generateKeypair, ecdh, derivePublicKey, createSession };
