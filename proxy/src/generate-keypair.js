const { generateKeypair, derivePublicKey } = require('./crypto');

const kp = generateKeypair();
const pubHex = kp.publicKey.toString('hex');

console.log('═══════════════════════════════════════════');
console.log('  E2E鍵ペアを生成しました');
console.log('═══════════════════════════════════════════');
console.log();
console.log('  公開鍵（マイナー用）:');
console.log('  ' + pubHex);
console.log();
console.log('  プロキシ起動コマンド:');
console.log('  node src/server.js');
console.log('═══════════════════════════════════════════');
