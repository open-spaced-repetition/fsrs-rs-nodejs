import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

const {
  FSRS,
  DEFAULT_PARAMETERS,
  FSRS5_DEFAULT_DECAY,
  FSRS6_DEFAULT_DECAY,
  FSRSReview,
  FSRSItem,
  MemoryState,
  NextStates,
  ItemState,
} = require('./index.cjs')

export {
  FSRS,
  DEFAULT_PARAMETERS,
  FSRS5_DEFAULT_DECAY,
  FSRS6_DEFAULT_DECAY,
  FSRSReview,
  FSRSItem,
  MemoryState,
  NextStates,
  ItemState,
}
