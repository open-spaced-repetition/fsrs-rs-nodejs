#![deny(clippy::all)]
use napi::JsNumber;
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
    Self::new(None)
  }
}
#[napi]
impl FSRS {
  #[napi(constructor)]
  pub fn new(parameters: Option<Vec<JsNumber>>) -> Self {
    let params: [f32; 19] = match parameters {
      Some(parameters) => {
        let mut array = [0.0; 19];
        for (i, value) in parameters.iter().enumerate().take(19) {
          array[i] = value.get_double().unwrap_or(0.0) as f32;
        }
        array
      }
      None => DEFAULT_PARAMETERS,
    };
    Self(fsrs::FSRS::new(Some(&params)).unwrap())
  }

  #[napi]
  pub fn compute_parameters(&self, train_set: Vec<&FSRSItem>) -> Vec<f32> {
    self
      .0
      .compute_parameters(train_set.iter().map(|x| x.0.clone()).collect(), None, true)
      .unwrap()
  }

  #[napi]
  pub fn next_states(
    &self,
    current_memory_state: Option<&MemoryState>,
    desired_retention: f64,
    days_elapsed: u32,
  ) -> NextStates {
    NextStates(
      self
        .0
        .next_states(
          current_memory_state.map(|x| x.0),
          desired_retention as f32,
          days_elapsed,
        )
        .unwrap(),
    )
  }

  #[napi]
  pub fn benchmark(&self, train_set: Vec<&FSRSItem>) -> Vec<f32> {
    self
      .0
      .benchmark(train_set.iter().map(|x| x.0.clone()).collect(), true)
  }

  #[napi]
  pub fn memory_state_from_sm2(
    &self,
    ease_factor: f64,
    interval: f64,
    sm2_retention: f64,
  ) -> MemoryState {
    MemoryState(
      self
        .0
        .memory_state_from_sm2(ease_factor as f32, interval as f32, sm2_retention as f32)
        .unwrap(),
    )
  }

  #[napi]
  pub fn memory_state(&self, item: &FSRSItem, starting_state: Option<&MemoryState>) -> MemoryState {
    MemoryState(
      self
        .0
        .memory_state(item.0.clone(), starting_state.map(|x| x.0))
        .unwrap(),
    )
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
  #[napi(getter)]
  pub fn rating(&self) -> u32 {
    self.0.rating
  }
  #[napi(getter)]
  pub fn delta_t(&self) -> u32 {
    self.0.delta_t
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
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

  #[napi(getter)]
  pub fn reviews(&self) -> Vec<FSRSReview> {
    self.0.reviews.iter().map(|x| FSRSReview(*x)).collect()
  }

  #[napi]
  pub fn long_term_review_cnt(&self) -> u32 {
    self.0.long_term_review_cnt() as u32
  }

  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "MemoryState")]
#[derive(Debug)]
pub struct MemoryState(fsrs::MemoryState);
#[napi]
impl MemoryState {
  #[napi(constructor)]
  pub fn new(stability: f64, difficulty: f64) -> Self {
    Self(fsrs::MemoryState {
      stability: stability as f32,
      difficulty: difficulty as f32,
    })
  }
  #[napi(getter)]
  pub fn stability(&self) -> f64 {
    self.0.stability as f64
  }
  #[napi(getter)]
  pub fn difficulty(&self) -> f64 {
    self.0.difficulty as f64
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "NextStates")]
#[derive(Debug)]
pub struct NextStates(fsrs::NextStates);
#[napi]
impl NextStates {
  #[napi(getter)]
  pub fn hard(&self) -> ItemState {
    ItemState(self.0.hard.clone())
  }
  #[napi(getter)]
  pub fn good(&self) -> ItemState {
    ItemState(self.0.good.clone())
  }
  #[napi(getter)]
  pub fn easy(&self) -> ItemState {
    ItemState(self.0.easy.clone())
  }
  #[napi(getter)]
  pub fn again(&self) -> ItemState {
    ItemState(self.0.again.clone())
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "ItemState")]
#[derive(Debug)]
pub struct ItemState(fsrs::ItemState);
#[napi]
impl ItemState {
  #[napi(getter)]
  pub fn memory(&self) -> MemoryState {
    MemoryState(self.0.memory)
  }
  #[napi(getter)]
  pub fn interval(&self) -> f32 {
    self.0.interval
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}
