'use strict';

const fs = require('node:fs');
const path = require('node:path');
const channel = require('./index.js');

const hex = (value) => Buffer.from(value).toString('hex');
const fixed = (byte) => Buffer.alloc(32, byte);
const client = channel.clientHello({
  sessionId: Buffer.alloc(16, 0x44),
  ephemeralPrivate: fixed(0x11),
});
const server = channel.serverHello(client, {
  identityPrivate: fixed(0x33),
  ephemeralPrivate: fixed(0x22),
});
const clientSession = channel.finishClient(client, server.frame);
const plaintext = Buffer.from('mining.notify vector', 'utf8');
const encrypted = clientSession.encrypt(plaintext);
const vector = {
  version: 1,
  session_id: hex(client.sessionId),
  client_ephemeral_private: hex(client.ephemeralPrivate),
  server_ephemeral_private: fixed(0x22).toString('hex'),
  server_identity_private: fixed(0x33).toString('hex'),
  client_hello: hex(client.frame),
  server_hello: hex(server.frame),
  plaintext: hex(plaintext),
  client_to_proxy_frame: hex(encrypted),
};
const output = path.resolve(__dirname, '../../../tests/crypto-vectors/v1.json');
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `${JSON.stringify(vector, null, 2)}\n`);
