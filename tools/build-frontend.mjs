import { build } from 'esbuild';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function readArg(name) {
  const index = process.argv.indexOf(name);
  if (index === -1) return '';
  return process.argv[index + 1] || '';
}

const outdirArg = readArg('--outdir');
if (!outdirArg) {
  console.error('[frontend-build] Missing required --outdir argument.');
  process.exit(1);
}

const outdir = path.resolve(projectRoot, outdirArg);
const entryPoints = [
  path.resolve(projectRoot, 'frontend/js/map.js'),
  path.resolve(projectRoot, 'frontend/js/recognize.js'),
  path.resolve(projectRoot, 'frontend/js/settings.js'),
];

await build({
  entryPoints,
  outdir,
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: ['es2020'],
  minify: true,
  sourcemap: false,
  charset: 'utf8',
  legalComments: 'none',
  logLevel: 'info',
  entryNames: '[name]',
  metafile: false,
});
