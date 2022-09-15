const assert = require('assert');
const { Value, MLP } = require('./index');

const n = new MLP(3, [4,4,1]);

const xs = [
  [2, 3, -1],
  [3, -1, 0.5],
  [0.5, 1, 1],
  [1, 1, -1],
];
const ys = [1, -1, -1, 1];

for (let i = 0; i < 100; i++) {
  // forward pass
  const ypred = xs.map(x => n.call(x));
  const loss = ypred.reduce((acc, cur, i) => acc.add(cur.sub(ys[i]).pow(2)), new Value(0));

  // backward pass
  for (let p in n.parameters()) {
    p.grad = 0;
  }
  loss.backward();

  // update
  for (let p of n.parameters()) {
    p.data += -0.01 * p.grad;
  }

  console.log(i, loss.data);
}

const ypred = xs.map(x => n.call(x));
for (let i in ypred) {
  console.log((Math.abs(ypred[i].data - ys[i])));
  assert(Math.abs(ypred[i].data - ys[i]) < 0.01);
}

const loss = ypred.reduce((acc, cur, i) => acc.add(cur.sub(ys[i]).pow(2)), new Value(0));
assert(loss.data < 0.01);

console.log('PASS');
