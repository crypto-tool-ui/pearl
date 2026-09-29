'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const channel = require('./index.js');

const vector = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/crypto-vectors/v1.json'), 'utf8'));
const fromHex = (value) => Buffer.from(value, 'hex');

test('fixed handshake and AES-GCM vector', () => {
  const client = channel.clientHello({
    sessionId: fromHex(vector.session_id),
    ephemeralPrivate: fromHex(vector.client_ephemeral_private),
  });
  assert.equal(client.frame.toString('hex'), vector.client_hello);
  const server = channel.serverHello(client, {
    identityPrivate: fromHex(vector.server_identity_private),
    ephemeralPrivate: fromHex(vector.server_ephemeral_private),
  });
  assert.equal(server.frame.toString('hex'), vector.server_hello);
  const clientSession = channel.finishClient(client, server.frame);
  const frame = clientSession.encrypt(fromHex(vector.plaintext));
  assert.equal(frame.toString('hex'), vector.client_to_proxy_frame);
  const proxySession = server.session;
  assert.deepEqual(proxySession.decrypt(frame), fromHex(vector.plaintext));
});

test('tamper, replay, wrong pin, wrong key, and truncation fail closed', () => {
  const client = channel.clientHello({ sessionId: fromHex(vector.session_id), ephemeralPrivate: fromHex(vector.client_ephemeral_private) });
  const server = channel.serverHello(client, { identityPrivate: fromHex(vector.server_identity_private), ephemeralPrivate: fromHex(vector.server_ephemeral_private) });
  const session = channel.finishClient(client, server.frame);
  const frame = session.encrypt(fromHex(vector.plaintext));
  const tampered = Buffer.from(frame);
  tampered[tampered.length - 1] ^= 1;
  assert.throws(() => server.session.decrypt(tampered), { code: 'authentication_failed' });
  assert.throws(() => server.session.decrypt(frame.subarray(0, frame.length - 1)), { code: 'invalid_length' });
  const replaySession = channel.serverHello(client, { identityPrivate: fromHex(vector.server_identity_private), ephemeralPrivate: fromHex(vector.server_ephemeral_private) }).session;
  replaySession.decrypt(frame);
  assert.throws(() => replaySession.decrypt(frame), { code: 'counter_mismatch' });
  const wrongKey = channel.serverHello(client, { identityPrivate: Buffer.alloc(32, 0x77), ephemeralPrivate: Buffer.alloc(32, 0x88) }).session;
  assert.throws(() => wrongKey.decrypt(frame), { code: 'authentication_failed' });
  const other = channel.clientHello({ sessionId: fromHex(vector.session_id), ephemeralPrivate: Buffer.alloc(32, 0x55), pinnedServerStatic: Buffer.alloc(32, 0x66) });
  assert.throws(() => channel.finishClient(other, server.frame), { code: 'pinned_key_mismatch' });
  session.sendCounter = channel.MAX_COUNTER;
  assert.throws(() => session.encrypt(Buffer.from('rollover')), { code: 'rekey_required' });
});
