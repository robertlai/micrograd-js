const engine = require('./src/engine');
const nn = require('./src/nn');

module.exports = {
  ...engine,
  ...nn,
};
