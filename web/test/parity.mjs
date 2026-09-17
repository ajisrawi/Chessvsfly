// node web/test/parity.mjs '["<fen>", ...]'  -> JSON list of fly evaluations
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Chess } from '../vendor/chess.js';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const FlyBrain = require('../flybrain.js');
const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, '..', 'data');
const meta = JSON.parse(await readFile(path.join(dataDir, 'fly_brain.json'), 'utf8'));
const buf = await readFile(path.join(dataDir, meta.bin));
const brain = new FlyBrain(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const fens = JSON.parse(process.argv[2]);
const out = fens.map((fen) => { const r = brain.evaluate(new Chess(fen)); return { fen, value: r.value, nActive: r.nActive }; });
console.log(JSON.stringify(out));
