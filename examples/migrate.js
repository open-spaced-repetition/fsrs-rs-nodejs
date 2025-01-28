import { FSRS, FSRSItem, FSRSReview } from '../index.js'

function migrateWithFullHistory() {
  // Create a new FSRS model
  const fsrs = new FSRS()

  // Simulate a full review history for a card
  const reviews = [new FSRSReview(3, 0), new FSRSReview(3, 1), new FSRSReview(4, 3), new FSRSReview(3, 7)]
  const item = new FSRSItem(reviews)

  // Calculate the current memory state
  const memoryState = fsrs.memoryState(item)
  console.log('Migrated memory state:', JSON.stringify(memoryState))
}

function migrateWithPartialHistory() {
  // Create a new FSRS model
  const fsrs = new FSRS()

  // Set the true retention of the original algorithm
  const sm2Retention = 0.9

  // Simulate the earliest card state from the first review log of Anki's card
  // - ease_factor: the ratio of the interval to the previous interval
  // - interval: the interval of the first review
  const easeFactor = 2.0
  const interval = 5.0

  // Calculate the earliest memory state
  const initialState = fsrs.memoryStateFromSm2(easeFactor, interval, sm2Retention)

  // Simulate partial review history
  const reviews = [new FSRSReview(3, 5), new FSRSReview(4, 10), new FSRSReview(3, 20)]
  const item = new FSRSItem(reviews)

  // Calculate the current memory state, passing the initial state
  const memoryState = fsrs.memoryState(item, initialState)
  console.log('Migrated memory state:', JSON.stringify(memoryState))
}

function migrateWithLatestState() {
  // Create a new FSRS model
  const fsrs = new FSRS()

  // Set the true retention of the original algorithm
  const sm2Retention = 0.9

  // Simulate the latest card state from Anki's card
  // - ease_factor: the ratio of the interval to the previous interval
  // - interval: the interval of the last review
  const easeFactor = 2.5
  const interval = 10.0

  // Calculate the memory state
  const memoryState = fsrs.memoryStateFromSm2(easeFactor, interval, sm2Retention)
  console.log('Migrated memory state:', JSON.stringify(memoryState))
}

function main() {
  console.log('Migrating with full history:')
  migrateWithFullHistory()
  console.log('\nMigrating with partial history:')
  migrateWithPartialHistory()
  console.log('\nMigrating with latest state only:')
  migrateWithLatestState()
}

main()
