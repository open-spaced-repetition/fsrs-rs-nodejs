#![deny(clippy::all)]
// https://github.com/rust-lang/rust-analyzer/issues/17429
use napi_derive::napi;

#[napi(js_name = "FSRS")]
#[derive(Debug)]
pub struct FSRS(fsrs::FSRS);
#[napi]
/// directly use fsrs::DEFAULT_PARAMETERS will cause error.
/// referencing statics in constants is unstable
/// see issue #119618 <https://github.com/rust-lang/rust/issues/119618> for more information
/// `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
/// to fix this, the value can be extracted to a `const` and then used.
pub const DEFAULT_PARAMETERS: [f32; 19] = [
  0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925,
  1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
];
impl Default for FSRS {
  fn default() -> Self {
    Self::new()
  }
}
#[napi]
impl FSRS {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self(fsrs::FSRS::new(Some(&[])).unwrap())
  }

  #[napi]
  pub fn compute_parameters(&self, train_set: Vec<&FSRSItem>) -> Vec<f32> {
    self
      .0
      .compute_parameters(train_set.iter().map(|x| x.0.clone()).collect(), None)
      .unwrap()
  }
}

#[napi(js_name = "FSRSReview")]
#[derive(Debug)]
pub struct FSRSReview(fsrs::FSRSReview);

#[napi]
impl FSRSReview {
  #[napi(constructor)]
  pub fn new(rating: u32, delta_t: u32) -> Self {
    Self(fsrs::FSRSReview { rating, delta_t })
  }
}

#[napi(js_name = "FSRSItem")]
#[derive(Debug)]
pub struct FSRSItem(fsrs::FSRSItem);
#[napi]
impl FSRSItem {
  #[napi(constructor)]
  pub fn new(reviews: Vec<&FSRSReview>) -> Self {
    Self(fsrs::FSRSItem {
      reviews: reviews.iter().map(|x| x.0).collect(),
    })
  }
  #[napi]
  pub fn long_term_review_cnt(&self) -> u32 {
    self.0.long_term_review_cnt() as u32
  }
}
