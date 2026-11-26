# fsrs-rs-nodejs

nodejs binding for [fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs)

## Online development

go to <https://idx.google.com/import>

## Development

```
yarn install
npm run build
find examples/ -exec node {} \;
```

Note: running `examples/train_csv.js` requires `revlog.csv` file, please download from
[revlog.csv](https://github.com/open-spaced-repetition/fsrs-rs/files/15046782/revlog.csv). Then put it in the root folder of this repository.

## FSRS-6 and FSRS-7

Since fsrs-rs 7, the binding supports two model versions. The number of parameters passed to `new FSRS()` selects the
version, and `fsrs.version` reports it as a `ModelVersion` (`'FSRS-6'` or `'FSRS-7'`):

| Parameters                         | Model  |
| ---------------------------------- | ------ |
| omitted, `null` or `[]`            | FSRS-7 |
| 34 values                          | FSRS-7 |
| 21 values (17 and 19 are expanded) | FSRS-6 |

The default parameters of both versions are exported, with the same values as fsrs-rs:

- `DEFAULT_PARAMETERS`: the FSRS-7 defaults (34 values), the same as `fsrs::DEFAULT_PARAMETERS`.
- `FSRS7_DEFAULT_PARAMETERS`: an alias of `DEFAULT_PARAMETERS`.
- `FSRS6_DEFAULT_PARAMETERS`: the FSRS-6 defaults (21 values), the same as `fsrs::FSRS6_DEFAULT_PARAMETERS`.

```js
import { FSRS, FSRS6_DEFAULT_PARAMETERS, MemoryState, ModelVersion, computeParameters } from 'fsrs-rs-nodejs'

const fsrs7 = new FSRS() // FSRS-7 defaults
const fsrs6 = new FSRS(FSRS6_DEFAULT_PARAMETERS)
console.log(fsrs7.version, fsrs6.version) // 'FSRS-7' 'FSRS-6'

// FSRS-7 tracks a fast stability next to the stability. It defaults to `stability`,
// so `new MemoryState(stability, difficulty)` keeps working.
const state = new MemoryState(10, 5, 8)
console.log(state.stabilityFast) // 8

// FSRS-7 accepts fractional elapsed days; FSRS-6 rounds them to whole days.
fsrs7.nextStates(state, 0.9, 2.5)
fsrs6.nextStates(state, 0.9, 2.5) // same as 3 days

// Training is a top-level function. It defaults to FSRS-7 (34 parameters);
// select FSRS-6 to get 21 parameters.
const fsrs7Parameters = await computeParameters(trainSet)
const fsrs6Parameters = await computeParameters(trainSet, { modelVersion: ModelVersion.Fsrs6 })
```

- `FSRSReview` accepts a fractional `deltaT`. FSRS-6 rounds it to whole days when updating memory states and when
  training.
- `computeParameters`, `benchmark` and `evaluateWithTimeSeriesSplits` take `modelVersion` (`ModelVersion` or
  `'FSRS-6'` / `'FSRS-7'`, default FSRS-7, like fsrs-rs) and `enableSchedPenalties` (FSRS-7 only, default `false`).
  All three are top-level functions, independent of any `FSRS` instance.
- `enableShortTerm` defaults to `true` and only applies to FSRS-6. FSRS-7 always enables short-term memory and ignores
  `enableShortTerm: false`, because it has no pre-training and would keep the default initial stability otherwise.
- Without a `trainingConfig`, FSRS-6 training keeps the fsrs-rs 6 hyperparameters (5 epochs, learning rate 0.04).
  In a `trainingConfig`, `numEpochs` and `learningRate` of 0 (the default) use the model version's defaults.
- `nextInterval(stability, ...)` assumes for FSRS-7 that the fast stability equals `stability`.

### Breaking changes in the fsrs-rs 7 upgrade

- `DEFAULT_PARAMETERS` now holds the 34 FSRS-7 defaults instead of the 21 FSRS-6 defaults. Use
  `FSRS6_DEFAULT_PARAMETERS` for the previous values.
- `new FSRS()`, `new FSRS([])` and `checkAndFillParameters([])` now use the FSRS-7 defaults.
- The `FSRS#computeParameters` and `FSRS#benchmark` instance methods are removed in favour of the top-level
  `computeParameters` and `benchmark` functions: `new FSRS().computeParameters(x, o)` becomes `computeParameters(x, o)`.
- `computeParameters`, `benchmark` and `evaluateWithTimeSeriesSplits` now train FSRS-7 by default, so
  `computeParameters` returns 34 parameters. Pass `modelVersion: ModelVersion.Fsrs6` for the previous behaviour.
- With FSRS-6 parameters, `nextStates` now rounds a fractional `daysElapsed` instead of truncating it. Integer values
  give the same results as before.
- The `toJSON()` strings of `MemoryState`, `ItemState` and `NextStates` include `stability_fast`, and the one of
  `FSRSReview` prints `delta_t` as a float.

---

# `@napi-rs/package-template`

![https://github.com/napi-rs/package-template/actions](https://github.com/napi-rs/package-template/workflows/CI/badge.svg)

> Template project for writing node packages with napi-rs.

# Usage

1. Click **Use this template**.
2. **Clone** your project.
3. Run `yarn install` to install dependencies.
4. Run `npx napi rename -n [name]` command under the project folder to rename your package.

## Install this test package

```
yarn add @napi-rs/package-template
```

## Support matrix

### Operating Systems

|                  | node14 | node16 | node18 |
| ---------------- | ------ | ------ | ------ |
| Windows x64      | ✓      | ✓      | ✓      |
| Windows x32      | ✓      | ✓      | ✓      |
| Windows arm64    | ✓      | ✓      | ✓      |
| macOS x64        | ✓      | ✓      | ✓      |
| macOS arm64      | ✓      | ✓      | ✓      |
| Linux x64 gnu    | ✓      | ✓      | ✓      |
| Linux x64 musl   | ✓      | ✓      | ✓      |
| Linux arm gnu    | ✓      | ✓      | ✓      |
| Linux arm64 gnu  | ✓      | ✓      | ✓      |
| Linux arm64 musl | ✓      | ✓      | ✓      |
| Android arm64    | ✓      | ✓      | ✓      |
| Android armv7    | ✓      | ✓      | ✓      |
| FreeBSD x64      | ✓      | ✓      | ✓      |

## Ability

### Build

After `yarn build/npm run build` command, you can see `package-template.[darwin|win32|linux].node` file in project root. This is the native addon built from [lib.rs](./src/lib.rs).

### Test

With [ava](https://github.com/avajs/ava), run `yarn test/npm run test` to testing native addon. You can also switch to another testing framework if you want.

### CI

With GitHub Actions, each commit and pull request will be built and tested automatically in [`node@14`, `node@16`, `@node18`] x [`macOS`, `Linux`, `Windows`] matrix. You will never be afraid of the native addon broken in these platforms.

### Release

Release native package is very difficult in old days. Native packages may ask developers who use it to install `build toolchain` like `gcc/llvm`, `node-gyp` or something more.

With `GitHub actions`, we can easily prebuild a `binary` for major platforms. And with `N-API`, we should never be afraid of **ABI Compatible**.

The other problem is how to deliver prebuild `binary` to users. Downloading it in `postinstall` script is a common way that most packages do it right now. The problem with this solution is it introduced many other packages to download binary that has not been used by `runtime codes`. The other problem is some users may not easily download the binary from `GitHub/CDN` if they are behind a private network (But in most cases, they have a private NPM mirror).

In this package, we choose a better way to solve this problem. We release different `npm packages` for different platforms. And add it to `optionalDependencies` before releasing the `Major` package to npm.

`NPM` will choose which native package should download from `registry` automatically. You can see [npm](./npm) dir for details. And you can also run `yarn add @napi-rs/package-template` to see how it works.

## Develop requirements

- Install the latest `Rust`
- Install `Node.js@10+` which fully supported `Node-API`
- Install `yarn@1.x`

## Test in local

- yarn
- yarn build
- yarn test

And you will see:

```bash
$ ava --verbose

  ✔ sync function from native code
  ✔ sleep function from native code (201ms)
  ─

  2 tests passed
✨  Done in 1.12s.
```

## Release package

Publishing uses npm trusted publishing (OIDC), so the workflow does not need a long-lived `NPM_TOKEN` secret. In the
npm settings for `fsrs-rs-nodejs` and every platform package under [`npm`](./npm), configure this trusted publisher:

- Provider: GitHub Actions
- Organization: `open-spaced-repetition`
- Repository: `fsrs-rs-nodejs`
- Workflow filename: `CI.yml`
- Allowed action: `npm publish`

Leave the environment name empty. The organization, repository, and workflow filename are case-sensitive.

When you want to release the package:

```
npm version [<newversion> | major | minor | patch | premajor | preminor | prepatch | prerelease [--preid=<prerelease-id>] | from-git]

git push
```

GitHub Actions builds and tests every native target, publishes versions that are not already on npm, and uploads the
native binaries to the matching GitHub release. Stable versions receive the npm `latest` tag; versions containing a
prerelease suffix receive the `next` tag. The CI workflow can also be run manually on `main` to retry a failed release.
