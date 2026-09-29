const path = require('path');

module.exports = {
  target: 'node',
  mode: 'production',
  entry: {
    stream: './src/stream.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js',
    clean: true,
  },
  node: {
    __dirname: false,
    __filename: false,
  },
};