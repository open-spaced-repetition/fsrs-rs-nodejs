#![deny(clippy::all)]
use napi::bindgen_prelude::{AsyncTask, JsFunction, Result};
use napi::{Env, JsNumber};
use std::sync::{Arc, Mutex};

mod train_task;
use train_task::{ComputeParametersTask, ProgressData};

// https://github.com/rust-lang/rust-analyzer/issues/17429
use napi_derive::napi;

#[napi(js_name = "FSRS")]
#[derive(Debug)]
pub struct FSRS(Arc<Mutex<fsrs::FSRS>>);
#[napi]
pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;
#[napi]
pub const FSRS6_DEFAULT_DECAY: f32 = 0.2;
#[napi]
/// directly use fsrs::DEFAULT_PARAMETERS will cause error.
/// referencing statics in constants is unstable
/// see issue #119618 <https://github.com/rust-lang/rust/issues/119618> for more information
/// `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
/// to fix this, the value can be extracted to a `const` and then used.
pub const DEFAULT_PARAMETERS: [f32; 21] = [
  0.2172,
  1.1771,
  3.2602,
  16.1507,
  7.0114,
  0.57,
  2.0966,
  0.0069,
  1.5261,
  0.112,
  1.0178,
  1.849,
  0.1133,
  0.3127,
  2.2934,
  0.2191,
  3.0004,
  0.7536,
  0.3332,
  0.1437,
  FSRS6_DEFAULT_DECAY,
];

impl Default for FSRS {
  fn default() -> Self {
    Self::new(None)
  }
}

#[napi]
impl FSRS {
  /// - Parameters must be provided before running commands that need them.
  /// - Parameters may be an empty array to use the default values instead.
  #[napi(constructor, catch_unwind)]
  pub fn new(parameters: Option<Vec<JsNumber>>) -> Self {
    let params: [f32; 21] = match parameters {
      Some(parameters) => {
        let mut array = [0.0; 21];
        for (i, value) in parameters.iter().enumerate().take(21) {
          array[i] = value.get_double().unwrap_or(0.0) as f32;
        }
        array
      }
      None => DEFAULT_PARAMETERS,
    };
    Self(Arc::new(Mutex::new(
      fsrs::FSRS::new(Some(&params)).unwrap(),
    )))
  }

  /// Calculate appropriate parameters for the provided review history.
  #[napi(ts_return_type = "Promise<Array<number>>")]
  pub fn compute_parameters(
    &self,
    train_set: Vec<&FSRSItem>,
    #[napi(ts_arg_type = "ComputeParametersOption")] options: Option<ComputeParametersOption>,
  ) -> Result<AsyncTask<ComputeParametersTask>> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    // Turn `JsFunction` into a `ThreadsafeFunction`
    let fn_form_js =
      if let Some(callback) = options.as_ref().and_then(|x| x.progress.as_ref()) {
        Some(callback.create_threadsafe_function(0, |ctx| {
          let progress_data: ProgressData = ctx.value;
          let env = ctx.env;
          let current = env.create_uint32(progress_data.current as u32)?;
          let total = env.create_uint32(progress_data.total as u32)?;
          let percent = env.create_double(progress_data.percent)?;
          let mut progress_obj = env.create_object()?;
          progress_obj.set_named_property("current", current)?;
          progress_obj.set_named_property("total", total)?;
          progress_obj.set_named_property("percent", percent)?;
          Ok(vec![progress_obj])
        })?)
      } else {
        None
      };

    let task = ComputeParametersTask {
      model: Arc::clone(&self.0),
      train_data,
      enable_short_term: options.as_ref().map_or(true, |x| x.enable_short_term),
      num_relearning_steps: options
        .as_ref()
        .and_then(|x| x.num_relearning_steps)
        .and_then(|x| x.get_int64().ok())
        .map(|x| x as usize),
      progress_callback: fn_form_js,
      progress_timeout: options
        .as_ref()
        .and_then(|x| x.timeout)
        .and_then(|x| x.get_int64().ok())
        .map_or(500, |x| x as u64),
    };

    Ok(AsyncTask::new(task))
  }

  /// The intervals and memory states for each answer button.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn next_states(
    &self,
    current_memory_state: Option<&MemoryState>,
    desired_retention: f64,
    days_elapsed: u32,
  ) -> NextStates {
    let locked_model = self.0.lock().unwrap();
    NextStates(
      locked_model
        .next_states(
          current_memory_state.map(|x| x.0),
          desired_retention as f32,
          days_elapsed,
        )
        .unwrap(),
    )
  }

  #[napi]
  pub fn benchmark(
    &self,
    train_set: Vec<&FSRSItem>,
    #[napi(ts_arg_type = "ComputeParametersOption")] options: Option<ComputeParametersOption>,
  ) -> Vec<f32> {
    let locked_model = self.0.lock().unwrap();
    locked_model.benchmark(fsrs::ComputeParametersInput {
      train_set: train_set.into_iter().map(|x| x.0.clone()).collect(),
      progress: None,
      enable_short_term: options
        .as_ref()
        .map(|x| x.enable_short_term)
        .unwrap_or(true),
      num_relearning_steps: options
        .as_ref()
        .and_then(|x| x.num_relearning_steps)
        .map(|x| x.get_int64().ok())
        .flatten()
        .map(|x| x as usize),
    })
  }

  /// Determine how well the model and parameters predict performance.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn evaluate(&self, env: Env, train_set: Vec<&FSRSItem>) -> Result<ModelEvaluation> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    let locked_model = self.0.lock().unwrap();
    let result = locked_model
      .evaluate(train_data, |_| true)
      .map_err(|e| napi::Error::from_reason(format!("FSRS evaluate failed: {}", e)))?;
    Ok(ModelEvaluation {
      log_loss: env.create_double(result.log_loss as f64)?,
      rmse_bins: env.create_double(result.rmse_bins as f64)?,
    })
  }

  /// If a card has incomplete learning history, memory state can be approximated from
  /// current sm2 values.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn memory_state_from_sm2(
    &self,
    ease_factor: f64,
    interval: f64,
    sm2_retention: f64,
  ) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
        .memory_state_from_sm2(ease_factor as f32, interval as f32, sm2_retention as f32)
        .unwrap(),
    )
  }

  /// Calculate the current memory state for a given card's history of reviews.
  /// In the case of truncated reviews, `startingState` can be set to the value of
  /// {@link memoryStateFromSm2} for the first review (which should not be included
  /// in {@link FSRSItem}). If not provided, the card starts as new.
  ///
  /// Parameters must have been provided when calling [`new FSRS()`]{@link constructor}.
  #[napi]
  pub fn memory_state(&self, item: &FSRSItem, starting_state: Option<&MemoryState>) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
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
  /// 1-4
  #[napi(getter)]
  pub fn rating(&self) -> u32 {
    self.0.rating
  }
  /// The number of days that passed
  /// # Warning
  /// `delta_t` for item first(initial) review must be 0
  #[napi(getter)]
  pub fn delta_t(&self) -> u32 {
    self.0.delta_t
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
///
/// When used during review, the last item should include the correct `delta_t`, but
/// the provided rating is ignored as all four ratings are returned by `.nextStates()`
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

#[napi(object)]
pub struct ModelEvaluation {
  pub log_loss: JsNumber,
  pub rmse_bins: JsNumber,
}

#[napi(object)]
pub struct ComputeParametersOption {
  /// Whether to enable short-term memory parameters
  pub enable_short_term: bool,
  /// Number of relearning steps
  pub num_relearning_steps: Option<JsNumber>,
  #[napi(
    ts_type = "(err: Error | null , value: { current: number, total: number, percent: number }) => void"
  )]
  pub progress: Option<JsFunction>,
  #[napi(ts_type = "number")]
  pub timeout: Option<JsNumber>,
}
