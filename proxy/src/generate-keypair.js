const { generateKeypair, derivePublicKey } = require('./crypto');

const kp = generateKeypair();
const pubHex = kp.publicKey.toString('hex');

console.log('═══════════════════════════════════════════');
console.log('  E2E Keypair generated');
console.log('═══════════════════════════════════════════');
console.log();
console.log('  Static pubkey (give to miners):');
console.log('  ' + pubHex);
console.log();
console.log('  Use this in proxy startup:');
console.log('  node src/server.js');
console.log('═══════════════════════════════════════════');
