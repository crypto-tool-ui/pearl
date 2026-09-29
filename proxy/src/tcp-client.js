/**
 * TCP client — connects to the backend Stratum pool.
 * Line-delimited JSON-RPC over plain TCP.
 */

const net = require('net');
const { EventEmitter } = require('events');

class TcpClient extends EventEmitter {
  constructor(host, port) {
    super();
    this.host = host;
    this.port = port;
    this.socket = null;
    this.buffer = '';
    this.connected = false;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.socket = new net.Socket();
      this.socket.setEncoding('utf8');
      this.buffer = '';

      const onConnect = () => {
        this.connected = true;
        this.socket.removeListener('error', onError);
        resolve();
      };

      const onError = (err) => {
        this.socket.removeListener('connect', onConnect);
        reject(err);
      };

      this.socket.once('connect', onConnect);
      this.socket.once('error', onError);
      this.socket.connect(this.port, this.host);

      // Line-delimited JSON-RPC
      this.socket.on('data', (data) => {
        this.buffer += data;
        let idx;
        while ((idx = this.buffer.indexOf('\n')) !== -1) {
          const line = this.buffer.substring(0, idx).trim();
          this.buffer = this.buffer.substring(idx + 1);
          if (line.length > 0) {
            this.emit('message', line);
          }
        }
      });

      this.socket.on('close', () => {
        this.connected = false;
        this.emit('close');
      });

      this.socket.on('error', (err) => {
        this.emit('error', err);
      });
    });
  }

  send(line) {
    if (!this.connected || !this.socket) throw new Error('TCP not connected');
    const framed = line.endsWith('\n') ? line : line + '\n';
    this.socket.write(framed);
  }

  close() {
    if (this.socket) {
      this.socket.destroy();
      this.socket = null;
    }
    this.connected = false;
  }
}

module.exports = TcpClient;
