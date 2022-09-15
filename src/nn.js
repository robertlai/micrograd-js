const { Value } = require('./engine');

class Neuron {
  constructor(nin) {
    this.w = [];
    for (let i = 0; i < nin; i++) {
      this.w.push(new Value(Math.random() * 2 - 1));
    }
    this.b = new Value(Math.random() * 2 - 1);
  }

  call(x) {
    const act = x.reduce((acc, cur, i) => acc.add(this.w[i].mul(cur)), this.b);
    const out = act.tanh();
    return out;
  }

  parameters() {
    return this.w.concat(this.b);
  }
}

class Layer {
  constructor(nin, nout) {
    this.neurons = [];
    for (let i = 0; i < nout; i++) {
      this.neurons.push(new Neuron(nin));
    }
  }

  call(x) {
    const outs = this.neurons.map(n => n.call(x));
    return outs.length === 1 ? outs[0] : outs;
  }

  parameters() {
    return [].concat(...this.neurons.map(n => n.parameters()));
  }
}

class MLP {
  constructor(nin, nouts) {
    const sz = [nin].concat(nouts);
    this.layers = [];
    for (let i = 0; i < nouts.length; i++) {
      this.layers.push(new Layer(sz[i], sz[i+1]));
    }
  }

  call(x) {
    this.layers.forEach(layer => x = layer.call(x));
    return x;
  }

  parameters() {
    return [].concat(...this.layers.map(l => l.parameters()));
  }
}

module.exports = {
  Neuron,
  Layer,
  MLP,
};
