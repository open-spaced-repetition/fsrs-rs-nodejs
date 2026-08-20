import { Bench } from 'tinybench'

import { DEFAULT_PARAMETERS, FSRS, MemoryState } from '../index.js'

const b = new Bench()
const fsrs = new FSRS(DEFAULT_PARAMETERS)
const memoryState = new MemoryState(7.0, 5.0)

b.add('FSRS.nextStates(new card)', () => {
  fsrs.nextStates(null, 0.9, 0)
})

b.add('FSRS.nextStates(existing card)', () => {
  fsrs.nextStates(memoryState, 0.9, 3)
})

await b.run()

console.table(b.table())
