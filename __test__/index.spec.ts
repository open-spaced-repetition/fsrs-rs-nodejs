import test from 'ava'
import type { TrainingConfig } from '../index.js'
import {
  DEFAULT_PARAMETERS,
  FSRS,
  FSRSItem,
  FSRSReview,
  checkAndFillParameters,
  defaultSimulatorConfig,
  evaluateWithTimeSeriesSplits,
  filterOutlier,
  simulate,
} from '../index.js'

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
  const parameters = checkAndFillParameters([])

  t.is(parameters.length, 21)
  t.deepEqual(parameters, DEFAULT_PARAMETERS)
})

test('runs a small simulation', (t) => {
  const config = defaultSimulatorConfig()
  config.deckSize = 20
  config.learnSpan = 20
  config.learnLimit = 2
  config.reviewLimit = 50

  const result = simulate(DEFAULT_PARAMETERS, 0.9, config, 1)

  t.is(result.reviewCntPerDay.length, 20)
  t.is(result.learnCntPerDay.length, 20)
  t.is(result.introducedCntPerDay.length, 20)
  t.true(result.costPerDay.every((cost) => Number.isFinite(cost)))
})

test('filters outliers and exposes time-series evaluation errors', (t) => {
  const item = new FSRSItem([new FSRSReview(3, 0), new FSRSReview(3, 1)])
  const result = filterOutlier([item], [item])

  t.true(Array.isArray(result.datasetForInitialization))
  t.true(Array.isArray(result.trainset))

  t.throws(() => evaluateWithTimeSeriesSplits([]))
})
