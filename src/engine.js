const assert = require('assert');

class Value {
  constructor(data, _children=[], _op='') {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(_children);
    this._op = _op;
  }

  add(other) {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + other.data, [this, other], '+');
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * other.data, [this, other], '*');
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  pow(other) {
    assert(!isNaN(other));
    const out = new Value(this.data ** other, [this], `**${other}`);
    out._backward = () => {
      this.grad += other * (this.data ** (other - 1)) * out.grad;
    };
    return out;
  }

  div(other) {
    other = other instanceof Value ? other : new Value(other);
    return this.mul(other.pow(-1));
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    other = other instanceof Value ? other : new Value(other);
    return this.add(other.neg());
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };
    return out;
  }

  exp() {
    const out = new Value(Math.exp(this.data), [this], 'exp');
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  backward() {
    const topo = [];
    const visited = new Set();
    function buildTopo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        for (let child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }
    buildTopo(this);
    topo.reverse();

    this.grad = 1;
    for (let node of topo) {
      node._backward();
    }
  }
}

module.exports = {
  Value,
};
