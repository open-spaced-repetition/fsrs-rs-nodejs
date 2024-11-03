const fs = require('fs');
const csv = require("@fast-csv/parse");
const { FSRSItem, FSRSReview, FSRS } = require('../index.js');

async function main() {
  // read revlog.csv
  // please download from
  // https://github.com/open-spaced-repetition/fsrs-rs/files/15046782/revlog.csv
  const content = fs.readFileSync('./revlog.csv');
  const records = await new Promise((resolve, reject) => {
    const results = [];
    csv.parseString(content, {
      headers: true,
      skipEmptyLines: true
    })
      .on('data', row => results.push(row))
      .on('error', error => reject(error))
      .on('end', () => resolve(results));
  });

  console.log(`revlogs.len() = ${records.length}`)
  console.time('full training time')

  // group by card_id
  const reviewsByCard = groupReviewsByCard(records);

  // convert to FSRSItems
  const fsrsItems = Object.values(reviewsByCard).flatMap(convertToFSRSItem);
  console.log(`fsrs_items.len() = ${fsrsItems.length}`);

  // create FSRS instance and optimize
  const fsrs = new FSRS(null);
  const optimizedParameters = fsrs.computeParameters(fsrsItems);
  console.log('optimized parameters:', optimizedParameters);
  console.timeEnd('full training time')
}

function groupReviewsByCard(records) {
  const reviewsByCard = {};

  records.forEach(record => {
    const cardId = record.card_id;
    if (!reviewsByCard[cardId]) {
      reviewsByCard[cardId] = [];
    }

    // convert unix timestamp (ms) to Date object
    // next day start at 4:00:00 UTC+8
    // use review_rating as rating
    const timestamp = parseInt(record.review_time);
    const date = new Date(timestamp);
    // convert to UTC+8 first
    date.setTime(date.getTime() + 8 * 60 * 60 * 1000);
    // then subtract 4 hours for next day cutoff
    date.setTime(date.getTime() - 4 * 60 * 60 * 1000);
    reviewsByCard[cardId].push([
      date,
      parseInt(record.review_rating)
    ]);
  });

  // ensure reviews for each card are sorted by time
  Object.values(reviewsByCard).forEach(reviews => {
    reviews.sort((a, b) => a[0] - b[0]);
  });

  return reviewsByCard;
}

function convertToFSRSItem(history) {
  const reviews = [];
  let lastDate = history[0][0];
  const items = [];

  for (const [date, rating] of history) {
    const deltaT = dateDiffInDays(lastDate, date);
    reviews.push(new FSRSReview(rating, deltaT));
    if (deltaT > 0) { // the last review is not the same day
      items.push(new FSRSItem([...reviews]));
    }
    lastDate = date;
  }

  return items.filter(item => item.longTermReviewCnt() > 0);
}

function dateDiffInDays(a, b) {
  const _MS_PER_DAY = 1000 * 60 * 60 * 24;
  const utc1 = Date.UTC(a.getUTCFullYear(), a.getUTCMonth(), a.getUTCDate());
  const utc2 = Date.UTC(b.getUTCFullYear(), b.getUTCMonth(), b.getUTCDate());
  return Math.floor((utc2 - utc1) / _MS_PER_DAY);
}

main();