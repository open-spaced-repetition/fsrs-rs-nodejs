//@ts-check
import { FSRS, MemoryState, DEFAULT_PARAMETERS } from '../index.js'

class Card {
  constructor() {
    this.due = new Date()
    this.memoryState = null
    this.scheduledDays = 0
    this.lastReview = null
  }
}

function scheduleNewCard() {
  // Create a new card
  const card = new Card()

  // Set desired retention
  const desiredRetention = 0.9

  // Create a new FSRS model
  const fsrs = new FSRS(DEFAULT_PARAMETERS)

  // Get next states for a new card
  const nextStates = fsrs.nextStates(card.memoryState, desiredRetention, 0)

  // Display the intervals for each rating
  console.log(`Again interval: ${Math.round(nextStates.again.interval * 10) / 10} days`)
  console.log(`Hard interval: ${Math.round(nextStates.hard.interval * 10) / 10} days`)
  console.log(`Good interval: ${Math.round(nextStates.good.interval * 10) / 10} days`)
  console.log(`Easy interval: ${Math.round(nextStates.easy.interval * 10) / 10} days`)

  // Assume the card was reviewed and the rating was 'good'
  const nextState = nextStates.good
  const interval = Math.max(1, Math.round(nextState.interval))

  // Update the card with the new memory state and interval
  card.memoryState = nextState.memory
  card.scheduledDays = interval
  card.lastReview = new Date()
  card.due = new Date(card.lastReview.getTime() + interval * 24 * 60 * 60 * 1000)

  console.log(`Next review due: ${card.due}`)
  console.log(`Memory state: ${JSON.stringify(card.memoryState)}`)
}

function scheduleExistingCard() {
  // Create an existing card with memory state and last review date
  const card = new Card()
  card.due = new Date()
  card.lastReview = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) // 7 days ago
  card.memoryState = new MemoryState(7.0, 5.0)
  card.scheduledDays = 7

  // Set desired retention
  const desiredRetention = 0.9

  // Create a new FSRS model
  const fsrs = new FSRS(DEFAULT_PARAMETERS)

  // Calculate the elapsed time since the last review
  const elapsedDays = Math.floor((Date.now() - card.lastReview.getTime()) / (24 * 60 * 60 * 1000))

  // Get next states for an existing card
  const nextStates = fsrs.nextStates(card.memoryState, desiredRetention, elapsedDays)

  // Display the intervals for each rating
  console.log(`Again interval: ${Math.round(nextStates.again.interval * 10) / 10} days`)
  console.log(`Hard interval: ${Math.round(nextStates.hard.interval * 10) / 10} days`)
  console.log(`Good interval: ${Math.round(nextStates.good.interval * 10) / 10} days`)
  console.log(`Easy interval: ${Math.round(nextStates.easy.interval * 10) / 10} days`)

  // Assume the card was reviewed and the rating was 'again'
  const nextState = nextStates.again
  const interval = Math.max(1, Math.round(nextState.interval))

  // Update the card with the new memory state and interval
  card.memoryState = nextState.memory
  card.scheduledDays = interval
  card.lastReview = new Date()
  card.due = new Date(card.lastReview.getTime() + interval * 24 * 60 * 60 * 1000)

  console.log(`Next review due: ${card.due}`)
  console.log(`Memory state: ${JSON.stringify(card.memoryState)}`)
}

console.log('Scheduling a new card:')
scheduleNewCard()

console.log('\nScheduling an existing card:')
scheduleExistingCard()
