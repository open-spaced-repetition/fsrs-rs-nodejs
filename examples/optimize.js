const { FSRSItem, FSRSReview, DEFAULT_PARAMETERS, FSRS } = require('../index')

function main() {
  // Create review histories for cards
  const reviewHistoriesOfCards = createReviewHistoriesForCards()
  // Convert review histories to FSRSItems
  const fsrsItems = reviewHistoriesOfCards.flatMap(convertToFSRSItem)
  console.log(`Size of FSRSItems: ${fsrsItems.length}`)

  // Create an FSRS instance with default parameters
  const fsrs = new FSRS(null) // Assuming you're not using any existing parameters
  console.log('Default parameters:', DEFAULT_PARAMETERS)

  // Optimize the FSRS model using the created items
  const optimizedParameters = fsrs.computeParameters(fsrsItems)
  console.log('Optimized parameters:', optimizedParameters)
}

function createReviewHistoriesForCards() {
  // This array represents a collection of review histories for multiple cards.
  // Each inner array represents the review history of a single card.
  // The structure is as follows:
  // - Outer array: Contains review histories for multiple cards
  // - Inner array: Represents the review history of a single card
  //   - Each element is an array: [Date, Rating]
  //     - Date: The date of the review (NaiveDate)
  //     - Rating: The rating given during the review (Number)
  //
  // The ratings typically follow this scale:
  // 1: Again, 2: Hard, 3: Good, 4: Easy
  //
  // This sample data includes various review patterns, such as:
  // - Cards with different numbers of reviews
  // - Various intervals between reviews
  // - Different rating patterns (e.g., consistently high, mixed, or improving over time)
  //
  // The data is then cycled and repeated to create a larger dataset of 100 cards.

  const reviewHistories = [
    [
      [new Date("2023-01-01"), 3],
      [new Date("2023-01-02"), 4],
      [new Date("2023-01-05"), 3],
      [new Date("2023-01-15"), 4],
      [new Date("2023-02-01"), 3],
      [new Date("2023-02-20"), 4],
    ],
    [
      [new Date("2023-01-01"), 2],
      [new Date("2023-01-02"), 3],
      [new Date("2023-01-04"), 4],
      [new Date("2023-01-12"), 3],
      [new Date("2023-01-28"), 4],
      [new Date("2023-02-15"), 3],
      [new Date("2023-03-05"), 4],
    ],
    [
      [new Date("2023-01-01"), 4],
      [new Date("2023-01-08"), 4],
      [new Date("2023-01-24"), 3],
      [new Date("2023-02-10"), 4],
      [new Date("2023-03-01"), 3],
    ],
    [
      [new Date("2023-01-01"), 1],
      [new Date("2023-01-02"), 1],
      [new Date("2023-01-03"), 3],
      [new Date("2023-01-06"), 4],
      [new Date("2023-01-16"), 4],
      [new Date("2023-02-01"), 3],
      [new Date("2023-02-20"), 4],
    ],
    [
      [new Date("2023-01-01"), 3],
      [new Date("2023-01-03"), 3],
      [new Date("2023-01-08"), 2],
      [new Date("2023-01-10"), 4],
      [new Date("2023-01-22"), 3],
      [new Date("2023-02-05"), 4],
      [new Date("2023-02-25"), 3],
    ],
    [
      [new Date("2023-01-01"), 4],
      [new Date("2023-01-09"), 3],
      [new Date("2023-01-19"), 4],
      [new Date("2023-02-05"), 3],
      [new Date("2023-02-25"), 4],
    ],
    [
      [new Date("2023-01-01"), 2],
      [new Date("2023-01-02"), 3],
      [new Date("2023-01-05"), 4],
      [new Date("2023-01-15"), 3],
      [new Date("2023-01-30"), 4],
      [new Date("2023-02-15"), 3],
      [new Date("2023-03-05"), 4],
    ],
    [
      [new Date("2023-01-01"), 3],
      [new Date("2023-01-04"), 4],
      [new Date("2023-01-14"), 4],
      [new Date("2023-02-01"), 3],
      [new Date("2023-02-20"), 4],
    ],
    [
      [new Date("2023-01-01"), 1],
      [new Date("2023-01-01"), 3],
      [new Date("2023-01-02"), 1],
      [new Date("2023-01-02"), 3],
      [new Date("2023-01-03"), 3],
      [new Date("2023-01-07"), 3],
      [new Date("2023-01-15"), 4],
      [new Date("2023-01-31"), 3],
      [new Date("2023-02-15"), 4],
      [new Date("2023-03-05"), 3],
    ],
    [
      [new Date("2023-01-01"), 4],
      [new Date("2023-01-10"), 3],
      [new Date("2023-01-20"), 4],
      [new Date("2023-02-05"), 4],
      [new Date("2023-02-25"), 3],
      [new Date("2023-03-15"), 4],
    ],
    [
      [new Date("2023-01-01"), 1],
      [new Date("2023-01-02"), 2],
      [new Date("2023-01-03"), 3],
      [new Date("2023-01-04"), 4],
      [new Date("2023-01-10"), 3],
      [new Date("2023-01-20"), 4],
      [new Date("2023-02-05"), 3],
      [new Date("2023-02-25"), 4],
    ],
    [
      [new Date("2023-01-01"), 3],
      [new Date("2023-01-05"), 4],
      [new Date("2023-01-15"), 3],
      [new Date("2023-01-30"), 4],
      [new Date("2023-02-15"), 3],
      [new Date("2023-03-05"), 4],
    ],
    [
      [new Date("2023-01-01"), 2],
      [new Date("2023-01-03"), 3],
      [new Date("2023-01-07"), 4],
      [new Date("2023-01-17"), 3],
      [new Date("2023-02-01"), 4],
      [new Date("2023-02-20"), 3],
      [new Date("2023-03-10"), 4],
    ],
    [
      [new Date("2023-01-01"), 4],
      [new Date("2023-01-12"), 3],
      [new Date("2023-01-25"), 4],
      [new Date("2023-02-10"), 3],
      [new Date("2023-03-01"), 4],
    ],
  ]
  // Cycle and repeat the array to create a larger dataset of 100 cards
  const res = []
  const len = reviewHistories.length
  for (let i = 0; i < 100; i++) {
    res.push(reviewHistories[i % len])
  }
  return res
}

/**
 * Converts a card's review history to an array of FSRSItems.
 *
 * @param {Array<[Date, Number]>} history - An array of reviews for a single card, where each element is a tuple of [Date, Rating].
 * @returns {Array<FSRSItem>} - An array of FSRSItems, each representing a single review in the history.
 */
function convertToFSRSItem(history) {
  const reviews = []
  let lastDate = history[0][0]
  const items = []


  for (const [date, rating] of history) {
    const deltaT = dateDiffInDays(lastDate, date)
    reviews.push(new FSRSReview(rating, deltaT))
    items.push(new FSRSItem([...reviews]))
    lastDate = date
  }

  return items.filter((item) => item.longTermReviewCnt() > 0)
}

function dateDiffInDays(a, b) {
  const _MS_PER_DAY = 1000 * 60 * 60 * 24;
  // Discard the time and time-zone information.
  const utc1 = Date.UTC(a.getFullYear(), a.getMonth(), a.getDate());
  const utc2 = Date.UTC(b.getFullYear(), b.getMonth(), b.getDate());

  return Math.floor((utc2 - utc1) / _MS_PER_DAY)
}

main()
