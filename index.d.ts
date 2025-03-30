/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

/**
 * directly use fsrs::DEFAULT_PARAMETERS will cause error.
 * referencing statics in constants is unstable
 * see issue #119618 <https://github.com/rust-lang/rust/issues/119618> for more information
 * `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
 * to fix this, the value can be extracted to a `const` and then used.
 */
export const DEFAULT_PARAMETERS: number[]
export interface ModelEvaluation {
  logLoss: number
  rmseBins: number
}
export declare class FSRS {
  /**
   * - Parameters must be provided before running commands that need them.
   * - Parameters may be an empty array to use the default values instead.
   */
  constructor(parameters?: Array<number> | undefined | null)
  /** Calculate appropriate parameters for the provided review history. */
  computeParameters(trainSet: Array<FSRSItem>, enableShortTerm: boolean, progressJsFn?: (err: null | Error, value: { current: number, total: number, percent: number }) => void, timeout?: number): Promise<Array<number>>
  /**
   * The intervals and memory states for each answer button.
   *
   * Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
   */
  nextStates(currentMemoryState: MemoryState | undefined | null, desiredRetention: number, daysElapsed: number): NextStates
  benchmark(trainSet: Array<FSRSItem>, enableShortTerm: boolean): Array<number>
  /**
   * Determine how well the model and parameters predict performance.
   *
   * Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
   */
  evaluate(trainSet: Array<FSRSItem>): ModelEvaluation
  /**
   * If a card has incomplete learning history, memory state can be approximated from
   * current sm2 values.
   *
   * Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
   */
  memoryStateFromSm2(easeFactor: number, interval: number, sm2Retention: number): MemoryState
  /**
   * Calculate the current memory state for a given card's history of reviews.
   * In the case of truncated reviews, `startingState` can be set to the value of
   * {@link memoryStateFromSm2} for the first review (which should not be included
   * in {@link FSRSItem}). If not provided, the card starts as new.
   *
   * Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
   */
  memoryState(item: FSRSItem, startingState?: MemoryState | undefined | null): MemoryState
}
export declare class FSRSReview {
  constructor(rating: number, deltaT: number)
  /** 1-4 */
  get rating(): number
  /**
   * The number of days that passed
   * # Warning
   * `delta_t` for item first(initial) review must be 0
   */
  get deltaT(): number
  toJSON(): string
}
/**
 * Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
 * to a single review, but contains the previous reviews of the card as well, after the
 * first one.
 *
 * When used during review, the last item should include the correct `delta_t`, but
 * the provided rating is ignored as all four ratings are returned by `.nextStates()`
 */
export declare class FSRSItem {
  constructor(reviews: Array<FSRSReview>)
  get reviews(): Array<FSRSReview>
  longTermReviewCnt(): number
  toJSON(): string
}
export declare class MemoryState {
  constructor(stability: number, difficulty: number)
  get stability(): number
  get difficulty(): number
  toJSON(): string
}
export declare class NextStates {
  get hard(): ItemState
  get good(): ItemState
  get easy(): ItemState
  get again(): ItemState
  toJSON(): string
}
export declare class ItemState {
  get memory(): MemoryState
  get interval(): number
  toJSON(): string
}
