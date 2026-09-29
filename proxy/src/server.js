/**
 * WebSocket ↔ TCP Bridge Proxy
 *
 * Uses native X25519 KeyObjects for E2E encryption.
 * Env: PROXY_PRIVATE_KEY (hex) — optional, auto-generated if omitted.
 *
 * FIX: Buffered WS→TCP forwarding to prevent silent message drop
 *      during TCP connection race condition.
 */

const { WebSocketServer } = require('ws');
const { generateKeypair, ecdh, derivePublicKey, createSession } = require('./crypto');
const TcpClient = require('./tcp-client');

const WS_PORT = parseInt(process.env.WS_PORT || '8443', 10);
const WS_HOST = process.env.WS_HOST || '0.0.0.0';
const TCP_HOST = process.env.TCP_TARGET_HOST || 'csd-us-east.lproute.com';
const TCP_PORT = parseInt(process.env.TCP_TARGET_PORT || '8760', 10);

// Generate or load static keypair
let proxyKeypair;
if (process.env.PROXY_PRIVATE_KEY) {
  proxyKeypair = generateKeypair(process.env.PROXY_PRIVATE_KEY);
  console.log(`[crypto] static pubkey: ${proxyKeypair.publicKey.toString('hex')}`);
} else {
  proxyKeypair = generateKeypair();
  console.log(`[crypto] no PROXY_PRIVATE_KEY set — using generated keypair`);
  console.log(`[crypto] static pubkey: ${proxyKeypair.publicKey.toString('hex')}`);
}

let totalConns = 0, activeConns = 0, totalMsgs = 0;

const wss = new WebSocketServer({ host: WS_HOST, port: WS_PORT });
console.log(`ws://${WS_HOST}:${WS_PORT} → tcp://${TCP_HOST}:${TCP_PORT}`);

wss.on('connection', (ws, req) => {
  totalConns++;
  activeConns++;
  const id = totalConns;
  console.log(`[#${id}] connection from ${req.socket.remoteAddress} (active: ${activeConns})`);

  let session = null, tcp = null, ready = false;
  let pendingQueue = []; // Buffer for WS→TCP messages before TCP is ready

  ws.on('message', (data, isBinary) => {
    // The WebSocket protocol is text-only. The first frame is the client's
    // 32-byte public key in hex; every later frame is ciphertext hex.
    if (isBinary || (!Buffer.isBuffer(data) && typeof data !== 'string')) return;
    const message = data.toString('utf8');

    if (!session) {
      console.log(`[#${id}] E2E handshake`);

      // ECDH: proxy's static secret + client's ephemeral pubkey
      try {
        if (!/^[0-9a-f]{64}$/i.test(message)) throw new Error('invalid client public key');
        const clientPub = Buffer.from(message, 'hex');
        const sharedSecret = ecdh(proxyKeypair.privateKeyObj, clientPub);
        session = createSession(sharedSecret, 'server');
      } catch (err) {
        console.error(`[#${id}] E2E handshake error: ${err.message}`);
        ws.close(1008, 'invalid handshake');
        return;
      }

      console.log(`[#${id}] E2E session established, connecting TCP backend...`);

      // Connect TCP backend
      tcp = new TcpClient(TCP_HOST, TCP_PORT);
      tcp.connect().then(() => {
        ready = true;
        console.log(`[#${id}] TCP connected, bridge active (flushing ${pendingQueue.length} queued messages)`);
        // Flush any messages that arrived before TCP was ready
        for (const plain of pendingQueue) {
          try {
            tcp.send(plain);
            totalMsgs++;
            console.log(`[#${id}] WS→TCP flushed: ${plain.substring(0, 80)}...`);
          } catch (err) {
            console.error(`[#${id}] WS→TCP flush error: ${err.message}`);
          }
        }
        pendingQueue = [];
      }).catch((err) => {
        console.error(`[#${id}] TCP failed: ${err.message}`);
        pendingQueue = [];
        ws.close(1011, 'backend unreachable');
      });

      tcp.on('message', (line) => {
        if (!session || ws.readyState !== 1) return;
        try {
          const enc = session.encrypt(Buffer.from(line, 'utf8'));
          ws.send(enc.toString('hex'));
          totalMsgs++;
          console.log(`[#${id}] TCP→WS: ${line.substring(0, 80)}...`);
        } catch (err) { console.error(`[#${id}] TCP→WS error: ${err.message}`); }
      });

      tcp.on('close', () => {
        console.log(`[#${id}] TCP disconnected`);
        if (ws.readyState === 1) ws.close(1011, 'backend closed');
      });
      tcp.on('error', (err) => console.error(`[#${id}] TCP error: ${err.message}`));

      // Reply with proxy's static public key AFTER setting up TCP
      // but we can send it now since TCP connect is async
      ws.send(proxyKeypair.publicKey.toString('hex'));

      return;
    }

    if (session) {
      try {
        if (!/^[0-9a-f]+$/i.test(message) || message.length % 2 !== 0) {
          throw new Error('invalid hexadecimal payload');
        }
        const wire = Buffer.from(message, 'hex');
        const plain = session.decrypt(wire).toString('utf8').trim();
        if (plain.length === 0) return;

        if (ready && tcp?.connected) {
          tcp.send(plain);
          totalMsgs++;
          console.log(`[#${id}] WS→TCP: ${plain.substring(0, 80)}...`);
        } else {
          // TCP not ready yet — queue the message
          pendingQueue.push(plain);
          console.log(`[#${id}] WS→TCP queued (${pendingQueue.length} pending): ${plain.substring(0, 80)}...`);
        }
      } catch (err) { console.error(`[#${id}] decrypt error: ${err.message}`); }
      return;
    }
  });

  ws.on('close', () => {
    activeConns--;
    console.log(`[#${id}] WS closed (active: ${activeConns})`);
    pendingQueue = [];
    if (tcp) tcp.close();
  });
  ws.on('error', (err) => console.error(`[#${id}] WS error: ${err.message}`));
});

setInterval(() => {
  if (activeConns > 0 || totalMsgs > 0)
    console.log(`[stats] active=${activeConns} total=${totalConns} msgs=${totalMsgs}`);
}, 60_000);

process.on('SIGINT', () => { console.log('\n[proxy] shutting down...'); wss.close(() => process.exit(0)); });
process.on('SIGTERM', () => { wss.close(() => process.exit(0)); });
