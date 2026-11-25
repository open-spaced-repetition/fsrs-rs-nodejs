import test from 'ava'
import type { NextStates, TrainingConfig } from '../index.js'
import {
  DEFAULT_PARAMETERS,
  FSRS,
  FSRS6_DEFAULT_DECAY,
  FSRS6_DEFAULT_PARAMETERS,
  FSRS7_DEFAULT_PARAMETERS,
  FSRSItem,
  FSRSReview,
  MemoryState,
  ModelVersion,
  benchmark,
  checkAndFillParameters,
  computeParameters,
  defaultSimulatorConfig,
  evaluateWithTimeSeriesSplits,
  filterOutlier,
  simulate,
} from '../index.js'

// Review histories from examples/optimize.js as [deltaT, rating] pairs.
// prettier-ignore
const REVIEW_HISTORIES: Array<Array<[number, number]>> = [
  [[0, 3], [1, 4], [3, 3], [10, 4], [17, 3], [19, 4]],
  [[0, 2], [1, 3], [2, 4], [8, 3], [16, 4], [18, 3], [18, 4]],
  [[0, 4], [7, 4], [16, 3], [17, 4], [19, 3]],
  [[0, 1], [1, 1], [1, 3], [3, 4], [10, 4], [16, 3], [19, 4]],
  [[0, 3], [2, 3], [5, 2], [2, 4], [12, 3], [14, 4], [20, 3]],
  [[0, 4], [8, 3], [10, 4], [17, 3], [20, 4]],
  [[0, 2], [1, 3], [3, 4], [10, 3], [15, 4], [16, 3], [18, 4]],
  [[0, 3], [3, 4], [10, 4], [18, 3], [19, 4]],
  [[0, 1], [0, 3], [1, 1], [0, 3], [1, 3], [4, 3], [8, 4], [16, 3], [15, 4], [18, 3]],
  [[0, 4], [9, 3], [10, 4], [16, 4], [20, 3], [18, 4]],
  [[0, 1], [1, 2], [1, 3], [1, 4], [6, 3], [10, 4], [16, 3], [20, 4]],
  [[0, 3], [4, 4], [10, 3], [15, 4], [16, 3], [18, 4]],
  [[0, 2], [2, 3], [4, 4], [10, 3], [15, 4], [19, 3], [18, 4]],
  [[0, 4], [11, 3], [13, 4], [16, 3], [19, 4]],
]

/** Builds the training set of examples/optimize.js: 100 cards cycling through the histories above. */
function createTrainSet(): FSRSItem[] {
  const items: FSRSItem[] = []
  for (let card = 0; card < 100; card++) {
    const reviews: FSRSReview[] = []
    for (const [deltaT, rating] of REVIEW_HISTORIES[card % REVIEW_HISTORIES.length]) {
      reviews.push(new FSRSReview(rating, deltaT))
      const item = new FSRSItem([...reviews])
      if (item.longTermReviewCnt() > 0) {
        items.push(item)
      }
    }
  }
  return items
}

function stabilities(nextStates: NextStates): number[] {
  return [nextStates.again, nextStates.hard, nextStates.good, nextStates.easy].map((state) => state.memory.stability)
}

function intervals(nextStates: NextStates): number[] {
  return [nextStates.again, nextStates.hard, nextStates.good, nextStates.easy].map((state) => state.interval)
}

test('schedules next states with default parameters', (t) => {
  const fsrs = new FSRS()
  const nextStates = fsrs.nextStates(null, 0.9, 0)

  for (const state of [nextStates.again, nextStates.hard, nextStates.good, nextStates.easy]) {
    t.true(Number.isFinite(state.interval))
    t.true(state.memory.stability > 0)
    t.true(state.memory.difficulty > 0)
  }
})

test('empty parameter array uses upstream defaults', (t) => {
  const defaultModel = new FSRS()
  const emptyParametersModel = new FSRS([])

  const defaultGood = defaultModel.nextStates(null, 0.9, 0).good
  const emptyGood = emptyParametersModel.nextStates(null, 0.9, 0).good

  t.is(emptyGood.interval, defaultGood.interval)
  t.is(emptyGood.memory.stability, defaultGood.memory.stability)
  t.is(emptyGood.memory.difficulty, defaultGood.memory.difficulty)
})

test('invalid parameter length throws', (t) => {
  t.throws(() => new FSRS([1, 2]), {
    message: /FSRS initialization failed/,
  })
})

test('calculates memory state from review history', (t) => {
  const fsrs = new FSRS()
  const item = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(4, 1)])
  const memoryState = fsrs.memoryState(item)

  t.true(memoryState.stability > 0)
  t.true(memoryState.difficulty > 0)
})

test('calculates interval and batched memory states', (t) => {
  const fsrs = new FSRS()
  const first = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(4, 1)])
  const second = new FSRSItem([new FSRSReview(2, 0), new FSRSReview(3, 2)])

  t.true(fsrs.nextInterval(null, 0.9, 3) > 0)

  const states = fsrs.memoryStateBatch([first, second])
  t.is(states.length, 2)
  t.true(states.every((state) => state.stability > 0))

  const historicalStates = fsrs.historicalMemoryStates(first)
  t.is(historicalStates.length, 2)

  const historicalBatch = fsrs.historicalMemoryStateBatch([first, second], [null, null])
  t.deepEqual(
    historicalBatch.map((row) => row.length),
    [2, 2],
  )
})

test('updates FSRSItem reviews', (t) => {
  const item = new FSRSItem([])
  item.reviews = [new FSRSReview(3, 0)]

  t.is(item.reviews.length, 1)
  t.is(item.reviews[0].rating, 3)
})

test('accepts and validates training config objects', (t) => {
  const config: TrainingConfig = { batchSize: 128 }

  t.is(config.batchSize, 128)

  t.throws(
    () => {
      evaluateWithTimeSeriesSplits([], { trainingConfig: { batchSize: 0 } })
    },
    { message: /batchSize/ },
  )
})

test('checks and fills parameters', (t) => {
  // fsrs 7 fills empty parameters with the FSRS-7 defaults (34 values).
  const parameters = checkAndFillParameters([])

  t.is(parameters.length, 34)
  t.deepEqual(parameters, DEFAULT_PARAMETERS)

  t.deepEqual(checkAndFillParameters(FSRS6_DEFAULT_PARAMETERS), FSRS6_DEFAULT_PARAMETERS)
  // Legacy 19-value parameters are expanded to 21 FSRS-6 values.
  t.is(checkAndFillParameters(FSRS6_DEFAULT_PARAMETERS.slice(0, 19)).length, 21)
  t.throws(() => checkAndFillParameters([1, 2]), { message: /checkAndFillParameters failed/ })
})

test('default parameter constants match the fsrs defaults', (t) => {
  t.is(DEFAULT_PARAMETERS.length, 34)
  t.is(FSRS6_DEFAULT_PARAMETERS.length, 21)
  t.deepEqual(FSRS7_DEFAULT_PARAMETERS, DEFAULT_PARAMETERS)
  t.is(FSRS6_DEFAULT_PARAMETERS[20], FSRS6_DEFAULT_DECAY)

  // `new FSRS()` uses the fsrs crate defaults, whose initial stabilities are w[0..4].
  t.deepEqual(stabilities(new FSRS().nextStates(null, 0.9, 0)), DEFAULT_PARAMETERS.slice(0, 4))
  t.deepEqual(
    stabilities(new FSRS(FSRS6_DEFAULT_PARAMETERS).nextStates(null, 0.9, 0)),
    FSRS6_DEFAULT_PARAMETERS.slice(0, 4),
  )
})

test('reports the model version selected by the parameters', (t) => {
  t.is(new FSRS().version, ModelVersion.Fsrs7)
  t.is(new FSRS([]).version, ModelVersion.Fsrs7)
  t.is(new FSRS(null).version, ModelVersion.Fsrs7)
  t.is(new FSRS(DEFAULT_PARAMETERS).version, ModelVersion.Fsrs7)
  t.is<string, string>(new FSRS(FSRS7_DEFAULT_PARAMETERS).version, 'FSRS-7')

  t.is(new FSRS(FSRS6_DEFAULT_PARAMETERS).version, ModelVersion.Fsrs6)
  t.is<string, string>(new FSRS(FSRS6_DEFAULT_PARAMETERS.slice(0, 19)).version, 'FSRS-6')
  t.is<string, string>(new FSRS(FSRS6_DEFAULT_PARAMETERS.slice(0, 17)).version, 'FSRS-6')

  t.throws(() => new FSRS(DEFAULT_PARAMETERS.slice(0, 33)), { message: /FSRS initialization failed/ })
})

test('schedules next states for FSRS-6 and FSRS-7', (t) => {
  for (const parameters of [FSRS6_DEFAULT_PARAMETERS, DEFAULT_PARAMETERS]) {
    const fsrs = new FSRS(parameters)
    const newCard = intervals(fsrs.nextStates(null, 0.9, 0))
    const reviewCard = intervals(fsrs.nextStates(new MemoryState(10, 5), 0.9, 10))

    for (const values of [newCard, reviewCard]) {
      t.true(values.every((interval) => Number.isFinite(interval) && interval > 0))
      t.true(values.every((interval, index) => index === 0 || interval > values[index - 1]))
    }
  }

  const fsrs6Good = new FSRS(FSRS6_DEFAULT_PARAMETERS).nextStates(null, 0.9, 0).good.memory
  t.is(fsrs6Good.stabilityFast, fsrs6Good.stability)

  // FSRS-7 starts the fast stability at 80% of the stability.
  const fsrs7Good = new FSRS().nextStates(null, 0.9, 0).good.memory
  t.true(Math.abs(fsrs7Good.stabilityFast - 0.8 * fsrs7Good.stability) < 1e-5)
})

test('rounds elapsed days for FSRS-6 and uses fractional days for FSRS-7', (t) => {
  const state = new MemoryState(10, 5)

  const fsrs6 = new FSRS(FSRS6_DEFAULT_PARAMETERS)
  t.deepEqual(intervals(fsrs6.nextStates(state, 0.9, 3.7)), intervals(fsrs6.nextStates(state, 0.9, 4)))
  t.notDeepEqual(intervals(fsrs6.nextStates(state, 0.9, 3.7)), intervals(fsrs6.nextStates(state, 0.9, 3)))

  const fsrs7 = new FSRS()
  t.notDeepEqual(intervals(fsrs7.nextStates(state, 0.9, 3.7)), intervals(fsrs7.nextStates(state, 0.9, 3)))
})

test('memory state supports the FSRS-7 fast stability', (t) => {
  const withoutFast = new MemoryState(10, 5)
  t.is(withoutFast.stability, 10)
  t.is(withoutFast.difficulty, 5)
  t.is(withoutFast.stabilityFast, 10)

  const withFast = new MemoryState(10, 5, 4)
  t.is(withFast.stability, 10)
  t.is(withFast.stabilityFast, 4)

  // FSRS-7 reads the fast stability, FSRS-6 ignores it.
  const fsrs7 = new FSRS()
  t.notDeepEqual(intervals(fsrs7.nextStates(withFast, 0.9, 5)), intervals(fsrs7.nextStates(withoutFast, 0.9, 5)))
  const fsrs6 = new FSRS(FSRS6_DEFAULT_PARAMETERS)
  t.deepEqual(intervals(fsrs6.nextStates(withFast, 0.9, 5)), intervals(fsrs6.nextStates(withoutFast, 0.9, 5)))
})

test('calculates memory states for FSRS-6 and FSRS-7', (t) => {
  const item = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(3, 1), new FSRSReview(4, 3), new FSRSReview(3, 7)])

  for (const parameters of [FSRS6_DEFAULT_PARAMETERS, DEFAULT_PARAMETERS]) {
    const fsrs = new FSRS(parameters)
    const memoryState = fsrs.memoryState(item)
    t.true(memoryState.stability > 0)
    t.true(memoryState.stabilityFast > 0)
    t.true(memoryState.difficulty > 0)

    const historicalStates = fsrs.historicalMemoryStates(item)
    t.is(historicalStates.length, 4)
    t.is(historicalStates[3].stability, memoryState.stability)
    t.is(historicalStates[3].stabilityFast, memoryState.stabilityFast)

    const sm2State = fsrs.memoryStateFromSm2(2.5, 10, 0.9)
    t.true(sm2State.stability > 0)
    const resumed = fsrs.memoryState(new FSRSItem([new FSRSReview(3, 10)]), sm2State)
    t.true(resumed.stability > 0)
  }

  // FSRS-7 maps SM-2 values to a state whose fast stability is 80% of the stability.
  const sm2State = new FSRS().memoryStateFromSm2(2.5, 10, 0.9)
  t.true(Math.abs(sm2State.stabilityFast - 0.8 * sm2State.stability) < 1e-4)
})

test('FSRSReview accepts fractional elapsed days', (t) => {
  t.is(new FSRSReview(3, 2).deltaT, 2)
  t.is(new FSRSReview(3, 0.5).deltaT, 0.5)

  const fractional = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(3, 2.5)])
  const whole = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(3, 2)])
  t.not(new FSRS().memoryState(fractional).stability, new FSRS().memoryState(whole).stability)
})

test('computes FSRS-6 parameters', async (t) => {
  const parameters = await computeParameters(createTrainSet(), {
    modelVersion: ModelVersion.Fsrs6,
  })

  t.is(parameters.length, 21)
  t.true(parameters.every(Number.isFinite))
  t.is(new FSRS(parameters).version, ModelVersion.Fsrs6)
})

test('computes FSRS-7 parameters by default and reports progress', async (t) => {
  const trainSet = createTrainSet()
  const progress: Array<{ current: number; total: number; percent: number }> = []
  const parameters = await computeParameters(trainSet, {
    progress: (err, value) => {
      if (!err) {
        progress.push(value)
      }
    },
    timeout: 10,
  })

  t.is(parameters.length, 34)
  t.true(parameters.every(Number.isFinite))
  t.notDeepEqual(parameters, DEFAULT_PARAMETERS)
  t.is(new FSRS(parameters).version, ModelVersion.Fsrs7)

  // Progress callbacks are queued from the worker thread; let them drain.
  await new Promise((resolve) => setTimeout(resolve, 50))
  t.true(progress.length > 0)
  t.true(progress.every((value) => value.current <= value.total))

  const withPenalties = await computeParameters(trainSet, {
    modelVersion: 'FSRS-7',
    enableSchedPenalties: true,
  })
  t.is(withPenalties.length, 34)
  t.true(withPenalties.every(Number.isFinite))
})

test('enableShortTerm false only applies to FSRS-6', async (t) => {
  const trainSet = createTrainSet()

  // FSRS-7 has no pre-training, so it always enables short-term memory and ignores `false`.
  const [fsrs7WithShortTerm, fsrs7WithoutShortTerm] = await Promise.all([
    computeParameters(trainSet, { enableShortTerm: true }),
    computeParameters(trainSet, { enableShortTerm: false }),
  ])
  t.deepEqual(fsrs7WithoutShortTerm, fsrs7WithShortTerm)
  t.notDeepEqual(fsrs7WithoutShortTerm.slice(0, 4), DEFAULT_PARAMETERS.slice(0, 4))

  const [fsrs6WithShortTerm, fsrs6WithoutShortTerm] = await Promise.all([
    computeParameters(trainSet, { modelVersion: ModelVersion.Fsrs6, enableShortTerm: true }),
    computeParameters(trainSet, { modelVersion: ModelVersion.Fsrs6, enableShortTerm: false }),
  ])
  t.notDeepEqual(fsrs6WithoutShortTerm, fsrs6WithShortTerm)
})

test('rejects misaligned card ids with a progress callback', async (t) => {
  await t.throwsAsync(
    computeParameters(createTrainSet(), {
      cardIds: [1],
      progress: () => {},
      timeout: 10,
    }),
  )
})

test('benchmarks FSRS-6 and FSRS-7 parameters', (t) => {
  const trainSet = createTrainSet()

  t.is(benchmark(trainSet, { modelVersion: ModelVersion.Fsrs6 }).length, 21)
  const fsrs7Parameters = benchmark(trainSet)
  t.is(fsrs7Parameters.length, 34)
  t.true(fsrs7Parameters.every(Number.isFinite))
})

test('training is not an FSRS instance method', (t) => {
  t.false('computeParameters' in FSRS.prototype)
  t.false('benchmark' in FSRS.prototype)
})

test('evaluates FSRS-6 and FSRS-7 models', (t) => {
  const trainSet = createTrainSet()

  for (const [parameters, modelVersion] of [
    [FSRS6_DEFAULT_PARAMETERS, ModelVersion.Fsrs6],
    [DEFAULT_PARAMETERS, ModelVersion.Fsrs7],
  ] as const) {
    const evaluation = new FSRS(parameters).evaluate(trainSet)
    t.true(Number.isFinite(evaluation.logLoss))
    t.true(Number.isFinite(evaluation.rmseBins))

    const splitEvaluation = evaluateWithTimeSeriesSplits(trainSet, { modelVersion })
    t.true(Number.isFinite(splitEvaluation.logLoss))
    t.true(Number.isFinite(splitEvaluation.rmseBins))
  }
})

test('runs a small simulation', (t) => {
  const config = defaultSimulatorConfig()
  config.deckSize = 20
  config.learnSpan = 20
  config.learnLimit = 2
  config.reviewLimit = 50

  for (const parameters of [DEFAULT_PARAMETERS, FSRS6_DEFAULT_PARAMETERS]) {
    const result = simulate(parameters, 0.9, config, 1)

    t.is(result.reviewCntPerDay.length, 20)
    t.is(result.learnCntPerDay.length, 20)
    t.is(result.introducedCntPerDay.length, 20)
    t.true(result.costPerDay.every((cost) => Number.isFinite(cost)))
    t.is(result.averageDesiredRetention, Math.fround(0.9))
  }

  config.learnLimit = 0
  t.is(simulate(DEFAULT_PARAMETERS, 0.9, config, 1).averageDesiredRetention, undefined)
})

test('filters outliers and exposes time-series evaluation errors', (t) => {
  const item = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(3, 1)])
  const result = filterOutlier([item], [item])

  t.true(Array.isArray(result.datasetForInitialization))
  t.true(Array.isArray(result.trainset))

  t.throws(() => evaluateWithTimeSeriesSplits([]))
})
